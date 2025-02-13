import os
import logging
from itertools import islice
from sentence_transformers import SentenceTransformer
from vector_db.vector_db import PineconeVector, VectorDB


EMBEDDING_MODEL='all-MiniLM-L6-v2'

class EmbeddingService:
    def __init__(self, vector_db: VectorDB):
        self.embedding_transformer = SentenceTransformer(EMBEDDING_MODEL)
        self.vector_db= vector_db

    def generate_embeddings(self, text: str):
        return self.embedding_transformer.encode([text])

    def batch_iterable(iterable, batch_size=200):
        """Yield successive batch_size chunks from an iterable."""
        iterator = iter(iterable)
        while batch := list(islice(iterator, batch_size)):
            yield batch

    def store_embeddings(self, documents: list[str]):
        """
        Reads the documents in batches and stores that batch embeddings at once.
        """
        for batch_id, batch in enumerate(self.batch_iterable(documents)):
            vectors=[]
            for chunk_id, chunk in batch:
                _embedding = self.generate_embeddings(chunk)
                _vector = PineconeVector(id=f'{batch_id}_{chunk_id}',
                            values=_embedding,
                            metadata={'text': chunk}
                            )
                vectors.append(_vector)
        self.vector_db.upsert_data(vectors)

    def extract_text_from_vectors(relevant_vectors: list):
        """
        Extracts text from each Pinecone Vector of the retrieved list of context vectors.
        """
        retrieved_context = []
        for _vector in relevant_vectors:
            _pc_vec = PineconeVector(_vector)
            retrieved_context.append(_pc_vec.metadata.text)
        return retrieved_context

    def retrieve_relevant_docs(self, query_text: str):
        """
        Fetches the relevant vectors.
        """
        query_embedding = self.generate_embeddings(query_text)
        relevant_vectors = self.vector_db.fetch_relevant_data(query_embedding)
        retrieved_context = self.extract_text_from_vectors(relevant_vectors)
        return retrieved_context


