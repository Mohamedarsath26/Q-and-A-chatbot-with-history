import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import create_retrival_chain,history_aware_retriever
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import MessagesPlaceholder,ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import openai
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

groq_api_key = os.getenv('GROQ_API_KEY')

st.title("Conversational RAG with pdf upload and chat history")

st.write("Updload Documents")

llm = ChatGroq(model='Gemma2-9b-It',api_key=groq_api_key)
embedding = OpenAIEmbeddings()

session_id = st.text_input("Session_Id",value='default_Session')

if 'store' not in st.session_state:
    st.session_state.store = {}

uploaded_file = st.file_uploader("Choose your file",type="pdf",accept_multiple_files=True)

if uploaded_file:
    document = []
    for upload in uploaded_file:
        temppdf = f"./temp.pdf"
        with open(temppdf,"wb") as file:
            file.write(upload.getvalue())
            file_name = upload.name

        loader = PyPDFDirectoryLoader(temppdf)
        docs = loader.load()
        document.append(docs)
    
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    splits= text_spliter.split_documents(documents=document)
    vector = FAISS.from_documents(documents=splits,embeddings = embedding)
    retriver = vector.as_retriver()
    








