from huggingface_hub import login
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough,RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
import gradio as gr
from dotenv import load_dotenv
import os

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

def get_retriever(filename):
    loader = PyPDFLoader(filename)
    document = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(document)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding= HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDING_MODEL")),
        collection_name="split_document"
    )

    return vectorstore.as_retriever()

def get_llm():
    return ChatOpenAI(
        model=os.getenv("MODEL_NAME"),
        base_url=os.getenv("BASE_URL"),
        api_key=os.getenv("API_KEY"),
        temperature=0
    )

def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system","""You are a kind and helpful assistant.
        When asked a question, answer straight to the point, using only the provided text context below.
        If the answer cannot be found in the context say "There is no such information provided in the document".
        Context : {context}
        """),
        ("human","{question}")
    ])

def generate_response(file,question):
    if file is None:
        return "Please provide a file"

    try :
        retriever = get_retriever(file.name)
        prompt = get_prompt()
        llm = get_llm()

        chain = RunnableMap(context=retriever, question=RunnablePassthrough()) | prompt | llm | StrOutputParser()

        return chain.invoke(question)

    except Exception as e:
        return f"Error processing the file : {e}"

with gr.Blocks(title="DOC RAG") as demo:
    gr.Markdown("<h1>DOC RAG PDF Q&A</h1>")
    gr.Markdown("<h2>Upload your document and ask questions</h2>")
    with gr.Row():
        with gr.Column():
            pdf = gr.File(file_types=[".pdf"],file_count="single",label="Upload PDF")
        with gr.Column():
            question = gr.Textbox(label="Question",placeholder="Type your question here...")
            submit = gr.Button("Ask")
        answer = gr.Textbox(label="Answer")
        submit.click(fn=generate_response,inputs=[pdf,question],outputs=[answer])

if __name__ == '__main__':
    demo.launch(theme=gr.themes.Base())

