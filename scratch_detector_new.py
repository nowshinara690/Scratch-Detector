from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ScratchRegion:
    """One detected scratch, with how much of its own local area is damaged."""

    box: tuple[int, int, int, int]  # x, y, width, height
    damage_percent: float  # % of this box's own area that is scratch ("crash")
    good_percent: float  # remaining % of this box's area that is undamaged ("valo")


@dataclass
class DetectionResult:
    mask: np.ndarray
    boxes: list[tuple[int, int, int, int]]
    regions: list[ScratchRegion]
    score: float
    area_percent: float  # % of the image (or of the analysed surface) covered by scratches


@dataclass
class FilterStages:
    """Intermediate images used in the spatial-filtering demonstration."""

    mean: np.ndarray
    median: np.ndarray
    laplacian: np.ndarray
    sobel: np.ndarray


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"Could not open image: {path}")

    return image


def normalise_filter_response(response: np.ndarray) -> np.ndarray:
    """Convert a signed or floating-point filter response into a display image."""

    magnitude = np.abs(response)

    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def create_filter_stages(image: np.ndarray) -> FilterStages:
    """Apply the required mean, median, Laplacian, and Sobel filters."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Smoothing stages remove small noise before edge extraction.
    mean = cv2.blur(gray, (5, 5))

    median = cv2.medianBlur(mean, 5)

    laplacian = normalise_filter_response(
        cv2.Laplacian(median, cv2.CV_32F, ksize=3)
    )

    sobel_x = cv2.Sobel(median, cv2.CV_32F, 1, 0, ksize=3)

    sobel_y = cv2.Sobel(median, cv2.CV_32F, 0, 1, ksize=3)

    sobel = normalise_filter_response(cv2.magnitude(sobel_x, sobel_y))

    return FilterStages(
        mean=mean, median=median, laplacian=laplacian, sobel=sobel
    )


# ============================================================
# 1. SHARPENING SPATIAL FILTER
# ============================================================


def sharpen(gray: np.ndarray) -> np.ndarray:
    """
    Sharpen the image using an unsharp spatial filter.
    This makes tiny scratches more visible.
    """

    blurred = cv2.GaussianBlur(gray, (0, 0), 2.0)

    sharpened = cv2.addWeighted(gray, 1.8, blurred, -0.8, 0)

    return sharpened


# ============================================================
# 2. DIRECTIONAL SPATIAL FILTER
# ============================================================


def line_kernel(length: int, angle_degrees: int) -> np.ndarray:

    if length % 2 == 0:
        length += 1

    kernel = np.zeros((length, length), dtype=np.uint8)

    center = length // 2
    radius = center

    angle = np.deg2rad(angle_degrees)

    dx = int(round(radius * np.cos(angle)))

    dy = int(round(radius * np.sin(angle)))

    cv2.line(kernel, (center - dx, center - dy), (center + dx, center + dy), 1, 1)

    return kernel


def spatial_scratch_response(
    image: np.ndarray, filter_stages: FilterStages | None = None
) -> np.ndarray:
    """
    Detect small bright/dark linear scratches
    using spatial-domain morphological filtering.
    """

    if filter_stages is None:
        filter_stages = create_filter_stages(image)

    # Use the mean + median filtered image as the noise-reduced base image.
    enhanced = sharpen(filter_stages.median)

    response = np.zeros_like(enhanced)

    # Detect scratches in multiple directions
    for angle in range(0, 180, 15):

        kernel = line_kernel(15, angle)

        top_hat = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel)

        black_hat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, kernel)

        response = cv2.max(response, top_hat)

        response = cv2.max(response, black_hat)

    # Local high-frequency texture
    background = cv2.GaussianBlur(enhanced, (0, 0), 5)

    high_frequency = cv2.absdiff(enhanced, background)

    response = cv2.max(response, high_frequency)

    return cv2.GaussianBlur(response, (3, 3), 0)


# ============================================================
# 3. HISTOGRAM SUBTRACTION
# ============================================================


def histogram_subtraction(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Compare input image and ideal/reference image
    using histogram subtraction.
    """

    if image.shape[:2] != reference.shape[:2]:

        reference = cv2.resize(
            reference, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA
        )

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)

    # Calculate histograms
    hist_image = cv2.calcHist([image_gray], [0], None, [256], [0, 256])

    hist_reference = cv2.calcHist([reference_gray], [0], None, [256], [0, 256])

    # Normalize
    cv2.normalize(hist_image, hist_image, 0, 1, cv2.NORM_MINMAX)

    cv2.normalize(hist_reference, hist_reference, 0, 1, cv2.NORM_MINMAX)

    # Histogram subtraction
    histogram_difference = cv2.absdiff(hist_image, hist_reference)

    # Convert histogram difference into
    # a scalar weight
    hist_score = float(np.mean(histogram_difference))

    weight = float(np.clip(hist_score * 20.0, 0.1, 1.0))

    # Local image difference
    image_gray = cv2.GaussianBlur(image_gray, (3, 3), 0)

    reference_gray = cv2.GaussianBlur(reference_gray, (3, 3), 0)

    local_difference = cv2.absdiff(image_gray, reference_gray)

    return cv2.convertScaleAbs(local_difference, alpha=weight)


# ============================================================
# 4. REFERENCE IMAGE ANALYSIS
# ============================================================


def reference_scratch_response(image: np.ndarray, reference: np.ndarray) -> np.ndarray:

    if image.shape[:2] != reference.shape[:2]:

        reference = cv2.resize(
            reference, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_AREA
        )

    image_blur = cv2.GaussianBlur(image, (3, 3), 0)

    reference_blur = cv2.GaussianBlur(reference, (3, 3), 0)

    # LAB color difference
    image_lab = cv2.cvtColor(image_blur, cv2.COLOR_BGR2LAB).astype(np.int16)

    reference_lab = cv2.cvtColor(reference_blur, cv2.COLOR_BGR2LAB).astype(np.int16)

    colour_difference = np.mean(np.abs(image_lab - reference_lab), axis=2).astype(
        np.uint8
    )

    # Spatial-domain local filtering
    background = cv2.GaussianBlur(colour_difference, (0, 0), 9)

    local_difference = cv2.absdiff(colour_difference, background)

    # Histogram subtraction
    hist_difference = histogram_subtraction(image, reference)

    # Combine both techniques
    response = cv2.addWeighted(local_difference, 0.75, hist_difference, 0.25, 0)

    return cv2.GaussianBlur(response, (3, 3), 0)


# ============================================================
# 5. SURFACE MASK
# ============================================================


def coloured_surface_mask(image: np.ndarray) -> np.ndarray:

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    colourful = ((hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 60)).astype(np.uint8) * 255

    count, labels, stats, _ = cv2.connectedComponentsWithStats(colourful)

    surface = np.zeros_like(colourful)

    min_area = image.shape[0] * image.shape[1] * 0.02

    for label in range(1, count):

        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            surface[labels == label] = 255

    # If not enough colour information,
    # use the complete image.
    if np.count_nonzero(surface) < image.shape[0] * image.shape[1] * 0.10:
        return np.full(image.shape[:2], 255, dtype=np.uint8)

    surface = cv2.morphologyEx(
        surface, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    )

    return cv2.dilate(surface, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))


# ============================================================
# 6. REMOVE NOISE
# ============================================================


def keep_line_like_pixels(mask: np.ndarray) -> np.ndarray:

    line_like = mask.copy()

    for angle in range(0, 180, 15):

        kernel = line_kernel(5, angle)

        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        line_like = cv2.max(line_like, opened)

    line_like = cv2.morphologyEx(
        line_like, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    return cv2.medianBlur(line_like, 3)


# ============================================================
# 7. SCRATCH DETECTION
# ============================================================


def detect_scratches(
    image: np.ndarray,
    reference: np.ndarray | None = None,
    filter_stages: FilterStages | None = None,
) -> DetectionResult:

    # Spatial filtering is ALWAYS performed
    spatial_response = spatial_scratch_response(image, filter_stages)

    if reference is not None:

        # Reference-based spatial + histogram analysis
        reference_response = reference_scratch_response(image, reference)

        # Combine spatial filtering with
        # reference/histogram analysis
        response = cv2.addWeighted(spatial_response, 0.55, reference_response, 0.45, 0)

        support = np.full(image.shape[:2], 255, dtype=np.uint8)

    else:

        # Without reference, spatial filtering
        # performs the detection.
        response = spatial_response

        support = coloured_surface_mask(image)

    valid_pixels = response[(response > 0) & (support > 0)]

    if valid_pixels.size == 0:

        empty = np.zeros(response.shape, dtype=np.uint8)

        return DetectionResult(
            mask=empty, boxes=[], regions=[], score=0.0, area_percent=0.0
        )

    # Adaptive threshold
    threshold = max(12.0, float(np.percentile(valid_pixels, 96)))

    candidate_mask = np.where((response >= threshold) & (support > 0), 255, 0).astype(
        np.uint8
    )

    mask = keep_line_like_pixels(candidate_mask)

    # Make thin scratches more visible
    mask = cv2.dilate(
        mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)

    boxes = []

    filtered_mask = np.zeros_like(mask)

    min_area = max(12, int(image.shape[0] * image.shape[1] * 0.00015))

    for label in range(1, count):

        x, y, width, height, area = stats[label]

        if area < min_area:
            continue

        pixels_y, pixels_x = np.where(labels == label)

        if len(pixels_x) < 3:
            continue

        points = np.column_stack((pixels_x, pixels_y)).astype(np.float32)

        direction_x, direction_y, _, _ = cv2.fitLine(
            points, cv2.DIST_L2, 0, 0.01, 0.01
        ).flatten()

        along = points[:, 0] * direction_x + points[:, 1] * direction_y

        across = points[:, 0] * -direction_y + points[:, 1] * direction_x

        elongation = (float(along.max() - along.min()) + 1.0) / max(
            1.0, float(across.max() - across.min()) + 1.0
        )

        if elongation < 1.35:
            continue

        if max(width, height) < 6:
            continue

        filtered_mask[labels == label] = 255

        padding = 3

        x1 = max(0, int(x) - padding)

        y1 = max(0, int(y) - padding)

        x2 = min(image.shape[1] - 1, int(x + width) + padding)

        y2 = min(image.shape[0] - 1, int(y + height) + padding)

        boxes.append((x1, y1, x2 - x1, y2 - y1))

    boxes.sort(key=lambda box: (box[1], box[0]))

    # ------------------------------------------------------------
    # Per-scratch damage percentage: for each detected region's own
    # bounding box, what fraction of that local box is scratch
    # ("crash") pixels versus the remaining undamaged ("valo") part.
    # ------------------------------------------------------------
    regions: list[ScratchRegion] = []

    for x, y, width, height in boxes:

        box_area = max(1, width * height)

        box_mask = filtered_mask[y : y + height, x : x + width]

        damaged_pixels = int(np.count_nonzero(box_mask))

        damage_percent = 100.0 * damaged_pixels / box_area

        good_percent = 100.0 - damage_percent

        regions.append(
            ScratchRegion(
                box=(x, y, width, height),
                damage_percent=damage_percent,
                good_percent=good_percent,
            )
        )

    if np.any(filtered_mask > 0):

        score = float(np.mean(response[filtered_mask > 0]))

    else:
        score = 0.0

    # ------------------------------------------------------------
    # Scratch coverage percentage.
    #
    # denominator = the surface actually being inspected:
    #   - with a reference image: the whole image
    #   - without a reference:    the coloured/relevant surface mask
    #     (falls back to the whole image if that mask is empty/trivial)
    # ------------------------------------------------------------
    total_pixels = int(image.shape[0] * image.shape[1])

    surface_pixels = int(np.count_nonzero(support))

    denominator = surface_pixels if surface_pixels > 0 else total_pixels

    scratch_pixels = int(np.count_nonzero(filtered_mask))

    area_percent = 100.0 * scratch_pixels / denominator if denominator > 0 else 0.0

    return DetectionResult(
        mask=filtered_mask,
        boxes=boxes,
        regions=regions,
        score=score,
        area_percent=area_percent,
    )


# ============================================================
# 8. FINAL RESULT IMAGE
# ============================================================


def annotate(image: np.ndarray, result: DetectionResult) -> np.ndarray:

    output = image.copy()

    # Red highlight over detected scratch areas
    overlay = output.copy()

    overlay[result.mask > 0] = (0, 0, 255)

    output = cv2.addWeighted(output, 0.75, overlay, 0.25, 0)

    # Green bounding boxes, each labelled with its own local damage %.
    for region in result.regions:

        x, y, width, height = region.box

        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 0), 2)

        label = f"{region.damage_percent:.0f}%"

        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1
        )

        label_y = max(label_height + baseline, y - 4)

        cv2.rectangle(
            output,
            (x, label_y - label_height - baseline),
            (x + label_width + 4, label_y + baseline),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            output,
            label,
            (x + 2, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    # Small text banner with the overall scratch coverage percentage.
    banner_text = f"Scratch area: {result.area_percent:.2f}%"

    (text_width, text_height), baseline = cv2.getTextSize(
        banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
    )

    cv2.rectangle(
        output,
        (5, 5),
        (5 + text_width + 10, 5 + text_height + baseline + 10),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        output,
        banner_text,
        (10, 10 + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output


# ============================================================
# 9. OUTPUT — COMPLETE FILTER PIPELINE
# ============================================================


def show_filter_pipeline(
    input_image: np.ndarray,
    filter_stages: FilterStages,
    detected_image: np.ndarray,
    scratch_count: int,
    area_percent: float,
    save_plot: Path | None = None,
) -> None:
    """Show the original image, four filter stages, and the final detection."""

    input_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

    detected_rgb = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)

    panels = (
        ("1. Original Image", input_rgb, None),
        ("2. Mean Filter (5 x 5)", filter_stages.mean, "gray"),
        ("3. Median Filter (5 x 5)", filter_stages.median, "gray"),
        ("4. Laplacian Filter", filter_stages.laplacian, "gray"),
        ("5. Sobel Filter", filter_stages.sobel, "gray"),
        (
            f"6. Final Scratch Detection ({scratch_count} region"
            f"{'s' if scratch_count != 1 else ''}, "
            f"{area_percent:.2f}% area)",
            detected_rgb,
            None,
        ),
    )

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for axis, (title, panel_image, colour_map) in zip(axes.flat, panels):
        axis.imshow(panel_image, cmap=colour_map)
        axis.set_title(title, fontweight="bold")
        axis.axis("off")

    fig.suptitle("Tiny Scratch Detector: Spatial Filter Results", fontsize=16)

    fig.tight_layout(rect=(0, 0, 1, 0.95))

    if save_plot:
        save_plot.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(save_plot, dpi=200, bbox_inches="tight")

        print(f"Filter plot saved to: {save_plot}")

    plt.show()


# ============================================================
# 10. COMMAND LINE
# ============================================================


def build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        description=(
            "Tiny Scratch Detector using "
            "spatial filtering and histogram subtraction."
        )
    )

    parser.add_argument(
        "input_image",
        nargs="?",
        type=Path,
        default=Path("image.png"),
        help="Input image.",
    )

    parser.add_argument(
        "--reference", type=Path, help=("Optional clean/ideal reference image.")
    )

    parser.add_argument(
        "--save", type=Path, help=("Optional path to save the final result.")
    )

    parser.add_argument(
        "--save-plot",
        type=Path,
        help=("Optional path to save the complete six-panel filter plot."),
    )

    return parser


# ============================================================
# 11. MAIN
# ============================================================


def main() -> None:

    args = build_parser().parse_args()

    image = load_image(args.input_image)

    reference = load_image(args.reference) if args.reference else None

    filter_stages = create_filter_stages(image)

    result = detect_scratches(image, reference, filter_stages)

    detected_image = annotate(image, result)

    print(
        f"Detected "
        f"{len(result.boxes)} "
        f"scratch region(s). "
        f"Score: {result.score:.1f}. "
        f"Scratched area: {result.area_percent:.2f}% "
        f"of the {'image' if reference is not None else 'analysed surface'}."
    )

    for index, region in enumerate(result.regions, start=1):

        x, y, width, height = region.box

        print(
            f"  Region {index} at (x={x}, y={y}, w={width}, h={height}): "
            f"damaged {region.damage_percent:.1f}% / "
            f"good {region.good_percent:.1f}% of that local area."
        )

    if args.save:

        args.save.parent.mkdir(parents=True, exist_ok=True)

        success = cv2.imwrite(str(args.save), detected_image)

        if not success:
            raise OSError(f"Could not save result image: " f"{args.save}")

        print(f"Result saved to: {args.save}")

    show_filter_pipeline(
        image,
        filter_stages,
        detected_image,
        len(result.boxes),
        result.area_percent,
        args.save_plot,
    )


if __name__ == "__main__":
    main()