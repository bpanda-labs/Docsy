import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from wrapper import ChatOpenRouter
from dotenv import load_dotenv
import os

load_dotenv()
open_router_key = os.getenv("OPENROUTER_API_KEY")

st.header("Docsy – Your PDF, Your Answers")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload your PDF", type="pdf")

if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # initializing embeddings
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # creating vector store FAISS
    vector_store = FAISS.from_texts(chunks, embedder)

    # get user question
    user_question = st.text_input("Type your question here")
    if user_question:
        matches = vector_store.similarity_search(user_question)

        # generate output
        llm = ChatOpenRouter(
            api_key=open_router_key,
            model="meta-llama/llama-3.3-70b-instruct:free",
        )

        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=matches, question=user_question)
        st.write(response)
