# Tiny Scratch Detector for Used Items

একটি local Streamlit web application, যা Classical Digital Image Processing ব্যবহার করে product image-এর সম্ভাব্য scratch, crack এবং surface defect শনাক্ত করে। Smartphone, laptop, car surface, furniture, metal, glass বা অন্যান্য product-এর JPG/PNG image সরাসরি upload করা যায়। Product category বা reference image প্রয়োজন নেই।

## প্রধান সুবিধা

- Direct JPG, JPEG এবং PNG upload
- Dark, minimal inspection dashboard
- Image preview এবং dimension display
- Detection sensitivity control
- Spatial filter kernel size control
- Canny lower/upper threshold control
- Grayscale conversion
- Gaussian, Median এবং Bilateral noise reduction
- Laplacian sharpening, Unsharp Masking এবং High-pass filtering
- Mean, Gaussian, Median এবং custom sharpening kernel
- Canny, Sobel এবং Laplacian edge detection
- GLCM texture analysis: Contrast, Homogeneity, Energy, Correlation
- Grayscale histogram analysis
- Automatic scratch candidate detection
- Green bounding box এবং local damage percentage
- Overall `Scratch area` percentage
- Binary defect mask
- Minor, Moderate এবং Severe rule-based classification
- Defect statistics table
- CSV export
- PDF inspection report
- SQLite inspection history
- Local upload এবং processed image storage

## Requirements

- Python 3.11 বা তার পরের version
- Windows, macOS অথবা Linux
- Internet শুধু প্রথমবার dependency install করার সময় প্রয়োজন

## Installation

Project folder-এ PowerShell খুলে চালান:

```powershell
cd "E:\From Desktop New\New folder"
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

যদি নিজের Python environment ব্যবহার করেন:

```powershell
python -m pip install -r requirements.txt
```

## Application চালানো

PowerShell:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Browser-এ খুলুন:

```text
http://localhost:8501
```

Windows-এ `run.bat` double-click করেও application চালু করা যায়।

Port পরিবর্তন করে চালাতে:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8502
```

## কীভাবে ব্যবহার করবেন

১. `Upload item image` অংশে একটি clear JPG বা PNG image দিন।

২. `Detection sensitivity` ঠিক করুন।

- 40–55: কম false positive, শুধু শক্তিশালী defect
- 60–75: সাধারণ balanced range
- 80–95: faint scratch ধরার চেষ্টা বেশি, false positive বাড়তে পারে

৩. Spatial filter kernel size নির্বাচন করুন। ৫ সাধারণ starting value।

৪. Canny lower এবং upper threshold প্রয়োজন অনুযায়ী adjust করুন।

৫. `Analyze Image` চাপুন।

৬. Main result image-এ green bounding box, percentage label এবং overall scratch area দেখুন।

৭. Expandable processing sections খুলে প্রতিটি filter-এর output দেখুন।

৮. CSV অথবা PDF download করুন। প্রয়োজন হলে `Save inspection to history` চাপুন।

## Processing pipeline

```text
Upload image
    -> Decode and validate
    -> Resize for efficient analysis
    -> Grayscale conversion
    -> Noise reduction
    -> Sharpening and local contrast
    -> Spatial filtering
    -> Edge detection
    -> Texture and histogram analysis
    -> Morphological line filtering
    -> Contour and Hough line analysis
    -> False-positive filtering
    -> Bounding boxes and severity
    -> Annotated image, CSV and PDF report
```

## Filters ও analysis

### Preprocessing

- **Gaussian Blur:** smooth weighted averaging; random noise কমায়।
- **Median Filter:** isolated bright/dark noise কমায় এবং ছোট detail preserve করার চেষ্টা করে।
- **Bilateral Filter:** noise কমানোর সঙ্গে edge preserve করার চেষ্টা করে।
- **Grayscale:** colour image-কে intensity image-এ convert করে।

### Sharpening

- **Laplacian Sharpening:** rapid intensity change ব্যবহার করে edge sharpen করে।
- **Unsharp Masking:** blurred image-এর সঙ্গে original difference যোগ করে detail prominent করে।
- **High-pass Filtering:** smooth background বাদ দিয়ে high-frequency detail দেখায়।

### Spatial filtering

- **Mean Filter:** neighbourhood average ব্যবহার করে smoothing করে।
- **Gaussian Filter:** weighted smoothing করে।
- **Median Filter:** neighbourhood median ব্যবহার করে।
- **Custom sharpening kernel:** 3×3 sharpening kernel ব্যবহার করে fine edge enhance করে।

### Edge detection

- **Canny:** thin edge map তৈরি করে; lower/upper threshold adjustable।
- **Sobel:** X এবং Y gradient combine করে edge strength দেখায়।
- **Laplacian:** second-order intensity change দেখায়।

### Texture analysis

Scikit-image-এর GLCM ব্যবহার করে global texture values তৈরি করা হয়:

- Contrast
- Homogeneity
- Energy
- Correlation

এগুলো পুরো image-এর texture summary; এগুলো একা scratch-এর location নির্ধারণ করে না।

### Histogram analysis

Grayscale image-এর intensity distribution graph দেখানো হয়। Histogram global comparison method; এটি একা scratch location বের করে না। বর্তমান application direct-image mode-এ কাজ করে এবং reference image প্রয়োজন করে না।

## Scratch detection

Detection module কয়েকটি response একত্র করে সম্ভাব্য scratch candidate তৈরি করে:

- Median denoising
- Unsharp-style enhancement
- Local contrast response
- Directional morphological top-hat response
- Directional black-hat response
- Canny edge response
- Bright-line response
- Colour বা dark-surface support mask
- Adaptive percentile threshold
- Morphological opening এবং closing
- Contour extraction
- Probabilistic Hough line fallback
- Area, length, aspect ratio এবং size filtering

Detected result-এর ওপর green bounding box আঁকা হয়। প্রতিটি box-এর পাশে local damage percentage থাকে। Main image-এর top-left-এ overall `Scratch area: X%` banner থাকে।

## Percentage-এর অর্থ

Local damage percentage:

```text
detected mask pixels / bounding box area × 100
```

Overall affected area:

```text
detected mask pixels / analysed image area × 100
```

এগুলো pixel-based image-processing estimate। এগুলো scratch-এর real depth, millimetre measurement, repair cost বা certified physical damage measurement নয়।

## Severity classification

প্রতিটি candidate-এর area, line length এবং contrast-এর rule-based score ব্যবহার করা হয়। Score অনুযায়ী candidate-কে Minor, Moderate অথবা Severe হিসেবে দেখানো হয়। এটি machine-learning confidence নয়।

## Export

### CSV

Defect table-এর তথ্য CSV হিসেবে download করা যায়। CSV-তে সাধারণত position, width, height, area, damage percentage, line length, contrast এবং severity থাকে।

### PDF

PDF report-এ থাকে:

- Project title
- Filename
- Product category হিসেবে All products
- Original image
- Annotated image
- Defect count
- Affected area percentage
- Severity distribution
- ব্যবহৃত processing techniques

## Local storage

```text
storage/
├── uploads/       uploaded image copies
├── processed/     annotated output images
├── reports/       report-related storage
└── database/
    └── inspections.db
```

SQLite history restart-এর পরেও থাকে। History tab থেকে saved inspection record দেখা, delete এবং clear করা যায়।

## Project structure

```text
app.py
requirements.txt
run.bat
README.md
PROJECT_DESCRIPTION_BN.txt
modules/
├── preprocessing.py
├── sharpening.py
├── spatial_filtering.py
├── edge_detection.py
├── texture_analysis.py
├── histogram_analysis.py
├── scratch_detection.py
├── severity_analysis.py
└── report_generator.py
database/
└── database.py
utils/
└── helpers.py
assets/
└── style.css
storage/
├── uploads/
├── processed/
├── reports/
└── database/
```

## ভালো result পাওয়ার পরামর্শ

- High-resolution image ব্যবহার করুন।
- Product surface close-up এবং focused রাখুন।
- পর্যাপ্ত আলো ব্যবহার করুন।
- Strong glare, shadow এবং reflection কম রাখুন।
- Plain বা কম textured background বেছে নিন।
- Scratch যেন visible contrast তৈরি করে এমন image ব্যবহার করুন।
- Sensitivity 60 থেকে শুরু করুন; প্রয়োজন অনুযায়ী বাড়ান বা কমান।
- বড় object-এর জন্য কয়েকটি close-up image আলাদাভাবে analyze করুন।

## Limitations

- Reflection, glare, shadow, texture এবং product boundary false positive তৈরি করতে পারে।
- খুব faint বা low-contrast scratch বাদ যেতে পারে।
- Intersecting crack একটি connected network তৈরি করলে একাধিক branch একসঙ্গে একটি region হতে পারে। Hough fallback আলাদা segment recover করার চেষ্টা করে।
- Generic detector সব ধরনের material-এর জন্য একই rule ব্যবহার করে; material-specific calibration নেই।
- Result rule-based candidate analysis; real-world physical measurement নয়।
- Reference image ছাড়া ideal-model histogram subtraction চালু থাকে না।

## Academic project summary

এই project একটি local web-based Digital Image Processing system, যা used product image থেকে সম্ভাব্য scratch এবং surface defect শনাক্ত করে। Image preprocessing, noise reduction, sharpening, spatial filtering, edge detection, texture analysis, morphology, contour analysis এবং Hough line detection একসঙ্গে ব্যবহার করে final annotated output তৈরি করা হয়। Result image-এ green bounding box, local damage percentage এবং overall scratch area দেখানো হয়। CSV এবং PDF export-এর মাধ্যমে result সংরক্ষণ করা যায় এবং SQLite history দিয়ে inspection record persistent রাখা হয়।

## Academic documentation

বিস্তারিত বাংলা project description-এর জন্য দেখুন:

`PROJECT_DESCRIPTION_BN.txt`

এই file-টি ChatGPT-তে দিয়ে academic project report বা PDF documentation তৈরি করা যাবে।
