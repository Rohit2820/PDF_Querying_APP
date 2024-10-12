from langchain_community.document_loaders import PyPDFLoader
import fitz  
from langchain.schema import Document

def load_pdf(uploaded_file):
    # Read the PDF file using PyMuPDF (fitz)
    pdf_reader = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    docs = []

    # Extract text from each page and store it as a Document
    for page_num in range(len(pdf_reader)):
        page = pdf_reader.load_page(page_num)
        text = page.get_text("text")
        docs.append(Document(page_content=text))
    
    return docs


