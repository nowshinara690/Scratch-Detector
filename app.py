from pathlib import Path
from datetime import datetime
import csv
from html import escape
from io import StringIO
import json
import streamlit as st
import matplotlib.pyplot as plt
import cv2

from modules.preprocessing import resize_for_analysis, grayscale, noise_reduction
from modules.sharpening import laplacian_sharpen, unsharp_mask, high_pass
from modules.spatial_filtering import outputs as spatial_outputs
from modules.edge_detection import outputs as edge_outputs
from modules.texture_analysis import analyze as texture_analyze
from modules.histogram_analysis import compare as histogram_compare
from modules.scratch_detection import detect
from modules.severity_analysis import classify
from modules.report_generator import make_pdf
from database.database import init_db, add_inspection, list_inspections, delete_inspection, clear_history
from utils.helpers import bytes_to_bgr, bgr_to_rgb, png_bytes

ROOT = Path(__file__).resolve().parent
APP_VERSION = '2026-11-15-fresh-upload-analysis-v2'
for p in ['storage/uploads','storage/processed','storage/reports','storage/database']: (ROOT/p).mkdir(parents=True, exist_ok=True)
init_db()
st.set_page_config(page_title='Tiny Scratch Detector', page_icon='🔎', layout='wide')
st.markdown((ROOT/'assets/style.css').read_text(encoding='utf-8'), unsafe_allow_html=True)
if st.session_state.get('_app_version') != APP_VERSION:
    st.session_state.pop('analysis', None)
    st.session_state['_app_version'] = APP_VERSION

def image_card(title, image, description='', key=None, gray=False):
    st.markdown(f'**{title}**  \n<span style="color:#667085">{description}</span>', unsafe_allow_html=True)
    st.image(image if gray else bgr_to_rgb(image), width=280)
    st.download_button('Download PNG', png_bytes(image), file_name=f'{(key or title).lower().replace(" ","_")}.png', mime='image/png', key=f'dl_{key or title}')

def run_analysis(image, reference, sensitivity, kernel, canny_low, canny_high):
    image = resize_for_analysis(image); gray=grayscale(image); noise=noise_reduction(gray)
    sharp={'Laplacian Sharpening':laplacian_sharpen(gray),'Unsharp Masking':unsharp_mask(gray),'High-Pass Filtering':high_pass(gray)}
    spatial=spatial_outputs(gray,kernel); edges=edge_outputs(gray,canny_low,canny_high)
    ref_gray=grayscale(reference) if reference is not None else None
    hist=histogram_compare(gray,ref_gray); texture=texture_analyze(gray)
    detection=detect(image,sensitivity); severity=classify(detection['defects'])
    return locals()

def render_table(columns, rows):
    headers = ''.join(f'<th>{escape(str(column))}</th>' for column in columns)
    body = ''.join(
        '<tr>' + ''.join(
            f'<td>{escape(str(row.get(column, "")))}</td>' for column in columns
        ) + '</tr>'
        for row in rows
    )
    st.markdown(
        f'<table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table>',
        unsafe_allow_html=True,
    )

st.markdown('<div class="hero"><h1>🔎 Tiny Scratch Detector</h1><p>Classical computer-vision inspection for used items</p></div>', unsafe_allow_html=True)
tab_new, tab_history = st.tabs(['New Inspection','Inspection History'])
with tab_new:
    left,right=st.columns([1.15, .85], gap='large')
    with left:
        uploaded=st.file_uploader('Upload item image',type=['jpg','jpeg','png'],key='uploaded')
        st.caption('Works directly with smartphones, laptops, car surfaces, furniture, tools, and other physical products.')
    with right:
        st.markdown('#### Analysis controls')
        sensitivity=st.slider('Detection sensitivity',10,95,60)
        kernel=st.slider('Spatial filter kernel size',3,15,5,step=2)
        canny_low=st.slider('Canny lower threshold',10,200,50)
        canny_high=st.slider('Canny upper threshold',50,300,150)
    if uploaded:
        try:
            input_bytes=uploaded.getvalue()
            upload_signature = (uploaded.name, uploaded.size)
            if st.session_state.get('_upload_signature') != upload_signature:
                st.session_state.pop('analysis', None)
                st.session_state['_upload_signature'] = upload_signature
            image=bytes_to_bgr(input_bytes)
            st.image(bgr_to_rgb(image),caption=f'{uploaded.name} — {image.shape[1]} × {image.shape[0]} px',width=420)
            if st.button('Analyze Image',type='primary',use_container_width=True):
                with st.spinner('Running image-processing pipeline...'):
                    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                    safe_name=Path(uploaded.name).name.replace(' ','_')
                    upload_path=ROOT/'storage'/'uploads'/f'{stamp}_{safe_name}'
                    upload_path.write_bytes(input_bytes)
                    st.session_state.analysis=run_analysis(image,None,sensitivity,kernel,canny_low,canny_high)
                    processed_path=ROOT/'storage'/'processed'/f'{stamp}_annotated.png'
                    cv2.imwrite(str(processed_path),st.session_state.analysis['detection']['annotated'])
                    st.session_state.storage_paths={'upload':str(upload_path.relative_to(ROOT)),'processed':str(processed_path.relative_to(ROOT))}
                    st.session_state.original=image; st.session_state.filename=uploaded.name
        except Exception as e: st.error(f'Could not read this image: {e}')
    if 'analysis' in st.session_state:
        a=st.session_state.analysis; d=a['detection']; s=a['severity'];
        st.divider(); st.subheader('Latest analysis summary')
        m1,m2,m3,m4=st.columns(4); m1.metric('Potential defects',len(d['defects'])); m2.metric('Affected area',f"{d['affected_percent']:.2f}%"); m3.metric('Minor',s['distribution']['Minor']); m4.metric('Moderate / Severe',s['distribution']['Moderate']+s['distribution']['Severe'])
        st.caption('These are rule-based image-processing candidates, not machine-learning confidence or calibrated physical measurements.')
        st.subheader('Detected damage on main image')
        preview_col, _ = st.columns([0.72, 0.28])
        with preview_col:
            st.image(bgr_to_rgb(d['annotated']), width=680, caption='Detected scratch candidates. Labels show local damaged area percentage.')
        if not d['defects']:
            st.info('No scratch candidates were detected. Try increasing detection sensitivity or using a closer, well-lit image.')
        if st.button('Save inspection to history'):
            rid=add_inspection('All products',st.session_state.get('filename',uploaded.name),len(d['defects']),d['affected_percent'],'Rule-based candidate analysis',{'defects':d['defects'],'distribution':s['distribution'],'storage_paths':st.session_state.get('storage_paths',{})}); st.success(f'Inspection saved as #{rid}')
        st.subheader('Processing results')
        with st.expander('Original and grayscale'):
            c1,c2=st.columns(2); c1.image(bgr_to_rgb(a['image']),caption='Original image',width=320); c2.image(a['gray'],caption='Grayscale conversion',width=320,clamp=True)
        with st.expander('Noise reduction'):
            cols=st.columns(3)
            for col,(name,img) in zip(cols,a['noise'].items()): col.image(img,caption=name,width=220,clamp=True)
        with st.expander('Sharpening spatial filters'):
            cols=st.columns(3)
            for col,(name,img) in zip(cols,a['sharp'].items()): col.image(img,caption=name,width=220,clamp=True)
        with st.expander('Spatial filters'):
            cols=st.columns(4)
            for col,(name,img) in zip(cols,a['spatial'].items()): col.image(img,caption=name,width=200,clamp=True)
        with st.expander('Edge detection'):
            cols=st.columns(3)
            for col,(name,img) in zip(cols,a['edges'].items()): col.image(img,caption=name,width=220,clamp=True)
        with st.expander('Texture analysis'):
            render_table(list(a['texture']), [a['texture']])
        with st.expander('Histogram analysis'):
            fig,ax=plt.subplots(figsize=(9,3)); ax.plot(a['hist']['current'],label='Uploaded image');
            if a['hist']['reference'] is not None: ax.plot(a['hist']['reference'],label='Reference image'); ax.legend()
            ax.set_title('Global grayscale histogram'); st.pyplot(fig); plt.close(fig)
            if a['hist']['correlation'] is not None: st.write({'Histogram correlation':a['hist']['correlation'],'Histogram distance':a['hist']['distance']}); st.caption('Histogram comparison is global; it does not independently locate scratches.')
        with st.expander('Detection details'):
            c1,c2=st.columns(2); c1.image(d['mask'],caption='Binary defect mask',width=320,clamp=True); c2.image(bgr_to_rgb(d['annotated']),caption='Annotated image with boxes and percentages',width=320)
            rows=[]
            for i,x in enumerate(s['defects'],1): rows.append({'Defect':i,**{k:x[k] for k in ['x','y','width','height','area','damage_percent','length','contrast']},'Severity':x['severity']})
            defect_columns=['Defect','x','y','width','height','area','damage_percent','length','contrast','Severity']
            render_table(defect_columns, rows)
        csv_rows=[{'defect':i+1,**x} for i,x in enumerate(s['defects'])]
        csv_output=StringIO(newline='')
        if csv_rows:
            csv_columns=list(dict.fromkeys(key for row in csv_rows for key in row))
            writer=csv.DictWriter(csv_output,fieldnames=csv_columns)
            writer.writeheader(); writer.writerows(csv_rows)
        csv_data=csv_output.getvalue().encode('utf-8')
        pdf=make_pdf('Tiny Scratch Detector Inspection','All products',st.session_state.get('filename',uploaded.name),bgr_to_rgb(a['image']),bgr_to_rgb(d['annotated']),{'count':len(d['defects']),'affected':d['affected_percent'],'distribution':s['distribution']},['Grayscale','Gaussian/Median/Bilateral','Laplacian/Unsharp/High-pass','Mean/Gaussian/Median','Canny/Sobel/Laplacian','GLCM','Morphological detection'])
        c1,c2=st.columns(2); c1.download_button('Download defect CSV',csv_data,'defects.csv','text/csv'); c2.download_button('Download PDF report',pdf,'inspection_report.pdf','application/pdf')
with tab_history:
    st.subheader('Saved inspections'); rows=list_inspections()
    if rows:
        history_columns=['ID','Date','Category','Filename','Defects','Affected %','Summary']
        history_rows=[dict(zip(history_columns, row)) for row in rows]
        render_table(history_columns, history_rows)
        selected=st.number_input('Inspection ID to delete',min_value=1,step=1,value=rows[0][0])
        c1,c2=st.columns(2)
        if c1.button('Delete selected record'): delete_inspection(int(selected)); st.rerun()
        if c2.button('Clear all history'): clear_history(); st.rerun()
    else: st.info('No saved inspections yet. Run an analysis and save it to history.')
