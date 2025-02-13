from abc import ABC, abstractmethod
from dotenv import load_dotenv
import os
import logging
from pinecone import Pinecone, ServerlessSpec

INDEX_NAME = 'RAG_chatbot'


class VectorDB(ABC):
    """
    Abstract class for vectorDB
    """
    @abstractmethod
    def create_index(self):
        raise NotImplementedError

    @abstractmethod
    def delete_index(self):
        raise NotImplementedError
    
    @abstractmethod
    def describe_index():
        raise NotImplementedError

    @abstractmethod
    def upsert_data():
        raise NotImplementedError

    @abstractmethod
    def fetch_data():
        raise NotImplementedError

    
class PineconeVector:
    id: str
    values: list[float]  # Embedding vector
    metadata: dict  # Additional metadata (like original text)

class PineconeDB(VectorDB): 
    """
    Concrete implementation of pinecone vectorDB.
    """
    def __init__(self):
        logging.info('Creating pinecone client')
        self.client = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.create_index(index_name=INDEX_NAME)

    def create_index(self, index_name: str):
        index_name=INDEX_NAME
        available_indexes = {index['name'] for index in self.client.list_indexes().indexes}
        if index_name not in available_indexes:
            logging.info(f'Create index : {index_name} in pinecone')
            self.client.create_index(name=index_name, metric="cosine", 
                                     dimension=384, 
                                     spec=ServerlessSpec(
                                         cloud='aws',
                                         region='us-east-1'
                                     ))
        else:
            logging.info(f'Index : {index_name} already exists')    
        self.index = self.client.Index(index_name)
        
    def delete_index(self, index_name: str):
        self.client.delete_index(name=index_name)
    
    def describe_index(self, index_name: str):
        self.client.describe_index(index_name)

    def upsert_data(self, vectors: list):
        """
        Insert/Updates the vectors in index
        """
        self.index.upsert(vectors=vectors)

    def fetch_relevant_data(self, query_vector, top_k=5, **kwargs):
        return self.index.query(
            vector=query_vector,
            top_k=top_k,
            metric="cosine",
            include_metadata=True,
            **kwargs
        )

    def fetch_data(self, ids):
        return self.index.fetch(ids=ids)
        







class VectorDBFactory:
    """Factory for creating vector database instances."""
    
    @staticmethod
    def get_vector_db(db_type: str):
        if db_type.lower() == "pinecone":
            return PineconeDB
        else:
            raise ValueError(f"Unsupported vector database: {db_type}")








#docsearch = self.client.from_documents(docs, embeddings, index_name=index_name)
# spec=ServerlessSpec(
#       cloud="aws",
#       region="us-east-1"
#     )


#Main
# v = VectorDBFactory().get_vector_db('pinecone')
# print(type(v))
# print(dir(v))