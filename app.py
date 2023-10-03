import streamlit as st
from streamlit_chat import message
from streamlit_Utilities import *
import openai

st.set_page_config(
    page_title="Home",
    page_icon="🐥"
)
st.sidebar.success("Select Above page")

openapi_key = st.secrets["open_ai_key"]
# openai.api_key = api_key.key
openai.api_key = openapi_key
SerpAPIWrapper.serp_api_key = st.secrets["serp_api_key"]


st.title('Content Generator Demo')
prompt = st.text_input('Write Your Topic.',key="input1")

st.sidebar.title("Enter your company name")
name=st.sidebar.text_input("Enter your Company name",key="input2")

st.sidebar.title("Enter your comapny type")
com_type=st.sidebar.text_input("Enter your compant type",key="input3")

st.sidebar.title("Enter Your company language")
language=st.sidebar.text_input("Enter your company language",key="input4")

st.sidebar.title("Enter your Company description")
description=st.sidebar.text_input("Enter your company description",key="input5")

st.sidebar.title("Upload your Company Deatils")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

st.sidebar.title("Enter Frequncy")
frequency=st.sidebar.text_input("Enter Your Frequncy",key="input6")
#comanpy name
# name="xevensolutions"
# #industry type
# com_type="Software Industry"
# #language
# language="English"
# #company description
# description=""""""
#details
# details=""""""


if 'instagram' not in st.session_state:
    st.session_state['instagram'] = []
if 'instagramImage' not in st.session_state:
    st.session_state['instagramImage'] = []
if 'twitter' not in st.session_state:
    st.session_state['twitter'] = []
if 'twitterImage' not in st.session_state:
    st.session_state['twitterImage'] = []
if 'facebook' not in st.session_state:
    st.session_state['facebook'] = []
if 'faceebokImage' not in st.session_state:
    st.session_state['faceebokImage'] = []
if 'linkedin' not in st.session_state:
    st.session_state['linkedin'] = []
if 'linkedinImage' not in st.session_state:
    st.session_state['linkedinImage'] = []
if 'similarity' not in st.session_state:
    st.session_state['similarity'] = []
if 'prompt' not in st.session_state:
    st.session_state['prompt'] = []

if (uploaded_file and prompt):
    if not st.session_state['prompt']:
        with open("uploaded_pdf.pdf", "wb") as pdf_file:
                pdf_file.write(uploaded_file.read())
        files_text = extract_text_from_pdf("uploaded_pdf.pdf")
        print(files_text)
        st.success("File loaded...")
        files_text=files_text+description
        print(files_text)
        # get text chunks
        text_chunks = get_text_chunks(files_text)
        st.success("file chunks created...")
        # create vetore stores
        vetorestore = get_vectorstore(text_chunks)
        st.success("Vectore Store Created...")
        st.session_state['similarity']=get_conversation_chain(vetorestore,prompt)
        multi_prompt=generate_multi_prompt_from_master(prompt,name,com_type,language,st.session_state['similarity'],frequency)
        st.session_state['prompt']=multi_prompt




insta_button, twitter_button, facebook_button, linkedIn_button = st.columns(4)

if st.session_state['prompt']:
    with insta_button:
        if st.sidebar.button("instagram", use_container_width=True):
            print("--------------------")
            print(len(st.session_state['prompt']))
            print("--------------------")
            print(name)
            st.session_state['instagram'] = generate_Instagram_multi_content(st.session_state['prompt'],name,com_type,language,st.session_state['similarity'])
            print(name)
            print(st.session_state['instagram'])
            instaRefineText=TextRefine(st.session_state['prompt'],name,com_type,language,st.session_state['similarity'])
            print(st.session_state['similarity'])
            print(instaRefineText)
            st.session_state['instagramImage']=generate_thumbnail_background(instaRefineText)
    with twitter_button:
        if st.sidebar.button("Twitter", use_container_width=True):
            print("--------------------")
            # st.session_state['similarity']=get_conversation_chain(vetorestore,prompt)
            print(st.session_state['prompt'])
            print("--------------------")
            st.session_state['twitter'] = generate_Twitter_multi_content(st.session_state['prompt'],name,com_type,language,st.session_state['similarity'])
            twitterRefineText=TextRefine(st.session_state['prompt'],name,com_type,language,st.session_state['similarity'])
            st.session_state['twitterImage']=generate_thumbnail_background(twitterRefineText)
    with facebook_button:
        if st.sidebar.button("Facebook", use_container_width=True):
            print("--------------------")
            print(st.session_state['prompt'])
            print("--------------------")
            text= generate_Facebook_multi_content(st.session_state['prompt'],name,com_type,language,st.session_state['similarity'])
            #removing left emojis
            st.session_state['facebook'] =remove_emojis(text)
            fbRefineText=TextRefine(st.session_state['prompt'],name,com_type,language,st.session_state['similarity'])
            st.session_state['faceebokImage']=generate_thumbnail_background(fbRefineText)
    with linkedIn_button:
        if st.sidebar.button("LinkedIn", use_container_width=True):
            print("--------------------")
            print(st.session_state['prompt'])
            print("--------------------")
            text=generate_LinkedIn_multi_content(st.session_state['prompt'],name,com_type,language,st.session_state['similarity'])
            st.session_state['linkedin'] =remove_emojis(text)
            print(st.session_state['linkedin'])
            linkedinRefineText=TextRefine(st.session_state['prompt'],name,com_type,language,st.session_state['similarity'])
            st.session_state['linkedinImage']=generate_thumbnail_background(linkedinRefineText)


if st.session_state['prompt']:
    st.header("Prompts")
    for i in range(0,len(st.session_state['prompt'])):
        message(st.session_state['prompt'][i])
if st.session_state['instagram']:
    st.header("Instagram Post Generated")
    # st.write("Instagram")
    for i in range(0,len(st.session_state['instagram'])):
        message(st.session_state['instagram'][i])
        st.image(st.session_state['instagramImage'][i],caption='Generated Image',use_column_width=True)
    # st.write(newRespone)
if st.session_state['twitter']:
    st.header("Twitter Post Generated")
    # st.write("Twitter")
    for i in range(0,len(st.session_state['twitter'])):
        message(st.session_state['twitter'][i])
        st.image(st.session_state['twitterImage'][i],caption='Generated Image',use_column_width=True)
    # st.write(newRespone)
if st.session_state['facebook']:
    st.header("Facebook Post Generated")
    # st.write("Facebook")
    for i in range(0,len(st.session_state['facebook'])):
        message(st.session_state['facebook'][i])
        st.image( st.session_state['faceebokImage'][i],caption='Generated Image',use_column_width=True)
    # st.write(newRespone)
if st.session_state['linkedin']:
    st.header("LinkedIn Post Generated")
    # st.write("LinkedIn")
    for i in range(0,len(st.session_state['linkedin'])):
        message(st.session_state['linkedin'][i])
        st.image(st.session_state['linkedinImage'][i],caption='Generated Image',use_column_width=True)



