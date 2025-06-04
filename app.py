import os
import openai
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv
import tempfile


load_dotenv()

def upload_files():
    uploaded_files = st.file_uploader("Upload the PDF files", accept_multiple_files=True)
    return uploaded_files


def main():
    filepath = os.getcwd()
    # Read the text file containing the API key
    with open(filepath + "/OpenAI_API_Key.txt", "r") as f:
         openai_api_key = ' '.join(f.readlines())

    # Update the OpenAI API key by updating the environment variable
    os.environ["OPENAI_API_KEY"] = openai_api_key
    openai.api_key = openai_api_key


    # Configure the page settings for the Streamlit app
    st.set_page_config(page_title="Chat with PDF")

    # Display the header for the Streamlit app
    st.header("LangChain RAG Application")

    # Allow users to upload a PDF file
    # pdf = st.file_uploader("Upload your PDF", type="pdf")
    pdfs = upload_files()

    # Check if a PDF file has been uploaded
    if pdfs is not None:
      for pdf in pdfs:

        # Save the uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf.read())
            tmp_file_path = tmp_file.name

        # Load PDF document
        loader = PyPDFLoader(tmp_file_path)
        docs = loader.load()
        for i, doc in enumerate(docs):
          doc.metadata['page_number'] = i + 1
          doc.metadata['source'] = os.path.basename(tmp_file_path)

        # Set up the text splitter for splitting texts into chunkss
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        

        # Split the extracted text into chunks for efficient processing
        chunks = splitter.split_documents(docs)

        # Create embeddings and build a knowledge base for the chunks.
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        vectordb = FAISS.from_documents(chunks, embedding_model)

      # Allow the user to input a question about the PDF
      user_question = st.text_input("Ask a question about your PDF")
      
      # Check if a user question has been entered.
      if user_question:

          # Perform similarity search on the knowledge base using the user's question
          #docs = vectordb.similarity_search(user_question)

          # Set up a question-answering chain
          llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2)
          qa_chain = RetrievalQA.from_chain_type(
             llm=llm,
             chain_type="stuff",
             retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5}),
             return_source_documents=True
           )

          # Generate a response to the user's question using the question-answering chain
          response = qa_chain.invoke({"query": user_question})

          # Display the generated response
          st.markdown("### 🧠 Answer")
          st.write(response["result"])

          


if __name__ == '__main__':
    main()
