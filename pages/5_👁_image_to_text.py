import streamlit as st
from streamlit_chat import message
from streamlit_Utilities import *
import uuid
st.set_page_config(
    page_title="Ryan_Image_to_text",
    page_icon="🐥"
)

openapi_key = st.secrets["open_ai_key"]
# openai.api_key = api_key.key
openai.api_key = openapi_key



if 'image_to_text' not in st.session_state:
    st.session_state['image_to_text'] = []
if 'caption' not in st.session_state:
    st.session_state['caption'] = []
if 'test' not in st.session_state:
    st.session_state['test'] = []

st.title("Upload Your Image Here")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

# st.title('Comment Replier')
# prompt = st.text_input('Write Your Topic.')


st.sidebar.title('user requirments')
user_prompt = st.sidebar.text_input('Write Your Topic.')

imageToText,BlogS,BlogC,BlogI,BlogL,BlogSE=st.columns(6)
if uploaded_file and user_prompt :
    with imageToText:
        if st.sidebar.button("Transcribe the Image"):
            # filename = uploaded_file.name
            # file_extension = os.path.splitext(filename)[1]
            save_path = "image.png"
            with open(save_path, "wb") as f:
                f.write(uploaded_file.read())
            print(save_path)
            st.success(f"Image saved locally as: {save_path}")
            print("2")
            st.session_state['image_to_text']=image_to_text(save_path)
            print(user_prompt)
            st.session_state['caption']=Image_to_text_transcribe(st.session_state['image_to_text'],user_prompt)
            print(user_prompt)
# if prompt:
#     with BlogS:
#         if st.sidebar.button("Generate prompt"):
#             st.session_state['test']=Image_to_text_transcribe(prompt,user_prompt)


if st.session_state['caption']:
    st.header("Caption")
    # st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    message(st.session_state['caption'])
if st.session_state['image_to_text']:
    st.header("Image to text")
    # st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    message(st.session_state['image_to_text'])
    # print(st.session_state['image_to_text'])
    # st.write(st.session_state['test'])



