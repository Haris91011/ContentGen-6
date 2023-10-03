import streamlit as st
from streamlit_chat import message
from streamlit_Utilities import *
st.set_page_config(
    page_title="Ryan_Blog",
    page_icon="🐥"
)

openapi_key = st.secrets["open_ai_key"]
# openai.api_key = api_key.key
openai.api_key = openapi_key
SerpAPIWrapper.serp_api_key = st.secrets["serp_api_key"]


#comanpy name
name="xevensolutions"
#industry type
com_type="Software Industry"
#language
language="English"
#company description
description=""""""
#details
details=""""""


if 'similar' not in st.session_state:
    st.session_state['similar'] = []
if 'blogP' not in st.session_state:
    st.session_state['blogP'] = []
if 'blogI' not in st.session_state:
    st.session_state['blogI'] = []
if 'blogT' not in st.session_state:
    st.session_state['blogT'] = []
if 'blogS' not in st.session_state:
    st.session_state['blogS'] = []
if 'blogC' not in st.session_state:
    st.session_state['blogC'] = []
if 'blogSE' not in st.session_state:
    st.session_state['blogSE'] = []
if 'blogL' not in st.session_state:
    st.session_state['blogL'] = []
if 'blogSa' not in st.session_state:
    st.session_state['blogSa'] = []


st.title('Blog Content Generator Demo')
user_prompt = st.text_input('Write Your Topic.',key="input1")

st.sidebar.title("Enter your company name")
name=st.sidebar.text_input("Enter your Company name",key="input2")

st.sidebar.title("Enter your comapny type")
com_type=st.sidebar.text_input("Enter your compant type",key="input3")

st.sidebar.title("Enter Your company language")
language=st.sidebar.text_input("Enter your company language",key="input4")

st.sidebar.title("Enter your Company description")
description=st.sidebar.text_input("Enter your company description",key="input5")

st.sidebar.title("Upload your Company description")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

st.sidebar.title('Enter your Title')
str_prompt = st.sidebar.text_input('Write Your Title:',key="input6")

if (uploaded_file and user_prompt):
    with open("uploaded_pdf.pdf", "wb") as pdf_file:
            pdf_file.write(uploaded_file.read())
    files_text = extract_text_from_pdf("uploaded_pdf.pdf")
    st.success("File loaded...")
    files_text=files_text+description
    print(files_text)
    # get text chunks
    text_chunks = get_text_chunks(files_text)
    st.success("file chunks created...")
    # create vetore stores
    vetorestore = get_vectorstore(text_chunks)
    st.success("Vectore Store Created...")



BlogT,BlogS,BlogC,BlogI,BlogL,BlogSE=st.columns(6)


if user_prompt:
    with BlogT:
        if st.button("Blog Title", use_container_width=True):
            st.session_state['similar']=get_conversation_chain(vetorestore,user_prompt)
            print(st.session_state['similar'])
            print(name)
            st.session_state['blogT']=blog_Multi_Title_Generator_new(user_prompt,name,com_type,language,st.session_state['similar'])
            print(st.session_state['blogT'])

    # global_pro=str_prompt
if str_prompt:
    with BlogS:
        if st.sidebar.button("Blog Structure", use_container_width=True):
            print("--------------------")
            st.session_state['similar']=get_conversation_chain(vetorestore,str_prompt)
            print(st.session_state['similar'])
            print("--------------------")
            print(str_prompt)
            st.session_state['blogS']=generate_Blog_Structure_new(str_prompt,name,com_type,language,st.session_state['similar'])
    with BlogC:
        if st.sidebar.button("Blog Content",use_container_width=True):
            print("--------------------")
            st.session_state['similar']=get_conversation_chain(vetorestore,str_prompt)
            print(st.session_state['similar'])
            print(st.session_state['blogS'])
            st.session_state['blogC']=generate_Blog_Content_new(str_prompt,st.session_state['blogS'],name,com_type,language,st.session_state['similar'])
    with BlogI:
        if st.sidebar.button("Blog Image", use_container_width=True):
            print(str_prompt)
            print(st.session_state['blogC'])
            print("-----------------------------------------------")
            blogRefineText=blogMultiPromptGenerator(str_prompt,st.session_state['blogC'])
            st.session_state['blogI']=generate_multi_thumbnail_background(blogRefineText)
    with BlogSE:
        if st.sidebar.button("Blog SEO", use_container_width=True):
            print("4")
            print(str_prompt)
            print("-----------------------------------------------")
            st.session_state['blogSE']=generate_Blog_SEO(str_prompt)
    with BlogL:
        if st.sidebar.button("Blog Links", use_container_width=True):
            print("5")
            print(str_prompt)
            print("-----------------------------------------------")
            blogLink=topic_generate(str_prompt)
            st.session_state['blogL']=blog_repo_links(blogLink)
if st.session_state['blogT']:
    st.header("Blog Title")
    for i in range(0,len(st.session_state['blogT'])):
        message(st.session_state['blogT'][i])
if st.session_state['blogS']:
    st.header("Blog Structure")
    message(st.session_state['blogS'])
if st.session_state['blogC']:
    st.header("Blog Content")
    print(len(st.session_state['blogC']))
    message(st.session_state['blogC'])
if st.session_state['blogI']:
    st.header("Blog Image Generated")
    # st.write("Blog Image")
    for i in range(0,3): 
        st.image(st.session_state['blogI'][i],caption='Generated Image',use_column_width=True)
if st.session_state['blogSE']:
    st.header("Blog SEO words Generated")
    # st.write("Blog SEO")
    message(st.session_state['blogSE'])
if st.session_state['blogL']:
    st.write("Blog Links")
    message(st.session_state['blogL'])