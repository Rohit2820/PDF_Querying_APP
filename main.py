import streamlit as st
from loader import load_pdf
# from inference import create_vectorstore
# from inference import answer_query
import os
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv 
load_dotenv()

# Function to create a vector store from the PDF content

# Function to answer the user's query
def answer_query(query, retriever):
    llm = ChatOpenAI(model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))  # API key loaded from .env
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use clear sentences keep the "
        "answer concise and detailed."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    results = rag_chain.invoke({"input": query})
    return results['answer']

def create_vectorstore(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '(?=>\. )', ' ', ''], 
        chunk_size=1000, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    return retriever

# Streamlit app
def main():
    st.title("PDF Querying App")

    # Upload PDF
    pdf_file = st.file_uploader("Upload a PDF file", type="pdf")
    if pdf_file is not None:
        # Load PDF and create vector store
        with st.spinner("Processing PDF..."):
            docs = load_pdf(pdf_file)
            retriever = create_vectorstore(docs)
        st.success("PDF processed successfully!")

        # Input query
        query = st.text_input("Ask a question about the PDF:")
        if query:
            # Answer query using RAG
            with st.spinner("Answering..."):
                answer = answer_query(query, retriever)
            st.write(f"**Answer:** {answer}")

if __name__ == "__main__":
    main()
