import os
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def clean_text(text):
    """Basic text cleaning function"""
    text = text.strip()
    text = text.replace("\n", " ")  # Normalize newlines
    return text

def chunk_documents(documents: list[str], chunk_size=384, chunk_overlap=15) -> list[Document]:
    """
        Splits long documents into smaller chunks.
        Ex: My embeddings are 384 dimensional vectors. So, chosen 384 chunksize with 15 words overlap
    """
    documents = [Document(page_content=page) for page in documents]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    documents = text_splitter.split_documents(documents)
    documents = list(map(documents, lambda x:x.page_content))
    return documents


def read_data(file_path: str) -> list[str]:
    """
    Returns a List of documents after reading the data from the file_path.
    """
    #file_path = "queen_of_mahismathi.pdf"
    extension =  file_path.rsplit('.',1)[-1]
    documents=[]
    if extension=='pdf':
        with open(file_path, 'rb') as f:
            reader = PdfReader(f)
            documents = [page.extract_text() for page in reader.pages[3:]]
    elif extension=='txt':
        with open(file_path, 'rb') as f:
            documents.append(f.read())
    return documents