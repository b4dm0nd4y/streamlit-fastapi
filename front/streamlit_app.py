import requests
import streamlit as st

st.title('Simple FastAPI application')

BASE_URL = 'http://158.160.168.111:8000'

tab1, tab2, tab3 = st.tabs(['Image', 'Text', 'Table'])

def main():
    with tab1:
        image = st.file_uploader('Classify an image', type=['jpg', 'jpeg'])
        if st.button('Classify', key='image') and image is not None:
            st.image(image)
            files = {'file': image.getvalue()}
            res = requests.post(
                f"{BASE_URL + '/clf_image'}", files=files
            ).json()
            st.write(
                f"Class name: {res['class_name']}, class index: {res['class_index']}"
            )
    
    with tab2:
        txt = st.text_input('Classify text')
        if st.button('Classify', key='text'):
            text = {'text': txt}
            res = requests.post(
                f"{BASE_URL + '/clf_text'}", json=text
            )
            st.write(res.json()['label'])
            st.write(res.json()['prob'])
            
    with tab3:
        st.write('Classify table data')
        with st.form('query_form'):
            feature1 = st.text_input('Input value 1', value='0.0')
            feature2 = st.text_input('Input value 2', value='0.0')
            
            submitted = st.form_submit_button('Classify')
            if submitted and feature1 and feature2:
                vector = {'feature1': feature1, 'feature2': feature2}
                res = requests.post(f"{BASE_URL + '/clf_table'}", json=vector)
                
                st.write('feature1', feature1, 'feature2', feature2)
                st.write(f"Predicted class is {res.json()['prediction']}")
                

if __name__ == '__main__':
    main()
