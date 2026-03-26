import dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

def get_vectorstore(filename):
    loader = PyPDFLoader(filename)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(document)

    vectorstore = Chroma.from_documents(documents=chunks,embedding= HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL")) ,collection_name="split document")

    return vectorstore.as_retriever()

def get_llm():
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        temperature=0
    )



if __name__ == '__main__':
    print('PyCharm')

