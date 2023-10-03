import streamlit as st
from streamlit_chat import message
from streamlit_Utilities import *
# import pyaudio
from pathlib import Path
import pathlib
import wave
from audio_recorder_streamlit import audio_recorder
openapi_key = st.secrets["open_ai_key"]
# openai.api_key = api_key.key
openai.api_key = openapi_key
st.set_page_config(
    page_title="Ryan_Speech_to_text",
    page_icon="🐥"
)
if 'similare' not in st.session_state:
    st.session_state['similare'] = []
if 'insta' not in st.session_state:
    st.session_state['insta'] = []
if 'instaImage' not in st.session_state:
    st.session_state['instaImage'] = []
if 'twit' not in st.session_state:
    st.session_state['twit'] = []
if 'twitImage' not in st.session_state:
    st.session_state['twitImage'] = []
if 'face' not in st.session_state:
    st.session_state['face'] = []
if 'faceImage' not in st.session_state:
    st.session_state['faceImage'] = []
if 'linke' not in st.session_state:
    st.session_state['linke'] = []
if 'linkeImage' not in st.session_state:
    st.session_state['linkeImage'] = []
if 'SpeechToText' not in st.session_state:
    st.session_state['SpeechToText']=[] 
if 'VoiceRecording' not in st.session_state:
    st.session_state['VoiceRecording']=[]

#comanpy name
# name="Bata"
# #industry type
# com_type="Foot wear"
# #language
# language="English"
# #company description
# description=""""""
#details
# details=""""""

st.title('Speech to text')

insta_button, twitter_button, facebook_button, linkedIn_button, blog_Title,blog_structure,blog_content,blog_image,blog_SEO,blog_Links = st.columns(10)
voice,col1,speechtoText,col3,col4=st.columns(5)

st.sidebar.title("Enter your company name")
name=st.sidebar.text_input("Enter your Company name",key="input2")

st.sidebar.title("Enter your comapny type")
com_type=st.sidebar.text_input("Enter your compant type",key="input3")

st.sidebar.title("Enter Your company language")
language=st.sidebar.text_input("Enter your company language",key="input4")

st.sidebar.title("Enter your Company description")
description=st.sidebar.text_input("Enter your company description",key="input5")

st.sidebar.title("Upload your Company Details")
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])

st.header("press to record audio")
audio_bytes = audio_recorder(
    sample_rate=41_000,
    pause_threshold=60.0,
    text="",
    recording_color="#e8b62c",
    neutral_color="#6aa36f",
    icon_size="6x",)
if audio_bytes:
    # audio=st.audio(audio_bytes, format="audio/wav")
    with open("test.wav", "wb") as f:
        f.write(audio_bytes)
        st.success("Audio saved successfully.")
    filename="test.wav"
    audio_file = open(str(filename), "rb")
    with st.spinner('Converting'):
        try:
            st.session_state['VoiceRecording']=speechToText(audio_file)          
        except Exception as e:
            print(e)
            st.warning("OpenAI API key Error. Replace your key.")
# with speechtoText:
#         uploaded_file=st.sidebar.file_uploader("Upload an audio file", type=["mp3", "wav", "ogg"])
#         if uploaded_file:
#             with open("uploaded_audio.wav", "wb") as f:
#                 f.write(uploaded_file.read())
#             filename="uploaded_audio.wav"
#             audio_file = open(str(filename), "rb")
#             with st.spinner('Converting'):
#                 try:
#                     st.session_state['SpeechToText']=speechToText(audio_file)
#                 except Exception as e:
#                     print(e)
#                     st.warning("OpenAI API key Error. Replace your key.")

if (uploaded_file and st.session_state['VoiceRecording']):
    if not st.session_state['similare']:
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

if (uploaded_file and st.session_state['VoiceRecording']):
    with insta_button:
        if st.sidebar.button("Instagram", use_container_width=True):
            print("--------------------")
            st.session_state['similare']=get_conversation_chain(vetorestore,st.session_state['VoiceRecording'])
            print(st.session_state['similare'])
            print("--------------------")
            st.session_state['insta'] = generate_Instagram_content_new(st.session_state['VoiceRecording'],name,com_type,language,st.session_state['similare'])
            print(st.session_state['insta'])
            instaRefineText=TextRefine(st.session_state['VoiceRecording'],name,com_type,language,st.session_state['similare'])
            print(instaRefineText)
            st.session_state['instaImage']=generate_thumbnail_background(instaRefineText)
    with twitter_button:
        if st.sidebar.button("Twitter", use_container_width=True):
            print("--------------------")
            st.session_state['similare']=get_conversation_chain(vetorestore,st.session_state['VoiceRecording'])
            print(st.session_state['similare'])
            print("--------------------")
            st.session_state['twit'] = generate_Twitter_content_new(st.session_state['VoiceRecording'],name,com_type,language,st.session_state['similare'])
            twiitwerRefineText=TextRefine(st.session_state['VoiceRecording'],name,com_type,language,st.session_state['similare'])
            print(twiitwerRefineText)
            st.session_state['twitImage']=generate_thumbnail_background(twiitwerRefineText)

    with facebook_button:
        if st.sidebar.button("Facebook", use_container_width=True):
            print("--------------------")
            st.session_state['similare']=get_conversation_chain(vetorestore,st.session_state['VoiceRecording'])
            print(st.session_state['similare'])
            print("--------------------")
            text= generate_Facebook_content_new(st.session_state['VoiceRecording'],name,com_type,language,st.session_state['similare'])
            #removing left emojis
            st.session_state['face'] =remove_emojis(text)
            fbRefineText=TextRefine(st.session_state['VoiceRecording'],name,com_type,language,st.session_state['similare'])
            st.session_state['faceImage']=generate_thumbnail_background(fbRefineText)
    with linkedIn_button:
        if st.sidebar.button("Linkedin", use_container_width=True):
            print("--------------------")
            st.session_state['similare']=get_conversation_chain(vetorestore,st.session_state['VoiceRecording'])
            print("--------------------")
            text=generate_LinkedIn_content_new(st.session_state['VoiceRecording'],name,com_type,language,st.session_state['similare'])
            st.session_state['linke'] =remove_emojis(text)
            linkedinRefineText=TextRefine(st.session_state['VoiceRecording'],name,com_type,language,st.session_state['similare'])
            st.session_state['linkeImage']=generate_thumbnail_background(linkedinRefineText)



if st.session_state['VoiceRecording']:
    st.header("Generated Voice")
    message(st.session_state['VoiceRecording'])
if st.session_state['SpeechToText']:
    st.header("Speech to Text")
    message(st.session_state['SpeechToText'])
if st.session_state['insta']:
    st.header("Instagram Post Generated")
    # st.write("Instagram")
    message(st.session_state['insta'])
    st.image(st.session_state['instaImage'],caption='Generated Image',use_column_width=True)
    # st.write(newRespone)
if st.session_state['twit']:
    st.header("Twitter Post Generated")
    # st.write("Twitter")
    message(st.session_state['twit'])
    st.image(st.session_state['twitImage'],caption='Generated Image',use_column_width=True)
    # st.write(newRespone)
if st.session_state['face']:
    st.header("Facebook Post Generated")
    # st.write("Facebook")
    message(st.session_state['face'])
    st.image( st.session_state['faceImage'],caption='Generated Image',use_column_width=True)
    # st.write(newRespone)
if st.session_state['linke']:
    st.header("LinkedIn Post Generated")
    # st.write("LinkedIn")
    message(st.session_state['linke'])
    st.image(st.session_state['linkeImage'],caption='Generated Image',use_column_width=True)



