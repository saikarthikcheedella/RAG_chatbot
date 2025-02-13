import os
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

model_name_or_path =  'sentence-transformers/all-MiniLM-L6-v2'
sentences= 'I"m good re!'
#embedding_transformer = SentenceTransformer(model_name_or_path=model_name_or_path)
#embeddings = embedding_transformer.encode(sentences)
#print("done", embeddings, embeddings.shape, type(embeddings))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=384,  # Adjust chunk size based on your model's limit
    chunk_overlap=20  # Ensure continuity between chunks
)

def chunk_documents(documents):
    """Splits long documents into smaller chunks."""
    print(len(documents))
    documents = [Document(page_content=page) for page in documents]
    return text_splitter.split_documents(documents)








reader = PdfReader("./knowledge_base/queen_of_mahismathi.pdf")
pages = reader.pages
pages = [page.extract_text() for page in pages[3:]]
pages
#for page in pages:
res = chunk_documents(pages)
print(res[0].page_content)




# 1) from text_processig.py read knowledge base
# 2) pass it to chunking and embeddins, get ready with all chunked documents 
# 3) store all of those embeddinggs
# this is one time job, so be careful to use offline v/s online flags.
# 4)Later stages, focus on setting up model (in dummy file)
# 5) create a llm chain for RAG.

