"""
Doc2Vec (Document Embeddings) Pipeline for Legal Text.
Corresponds to Experiment 2 and Experiment 3 in Iyer (2020).
Maps variable-length legal decisions into dense, fixed-size semantic vectors.
"""

import os
from typing import List, Tuple, Dict, Any, Union
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from src.data_loader import simple_tokenize


class LegalDoc2VecPipeline:
    def __init__(
        self,
        vector_size: int = 150,
        window: int = 8,
        min_count: int = 5,
        epochs: int = 20,
        dm: int = 1, # 1 for Distributed Memory (PV-DM), 0 for Distributed Bag of Words (PV-DBOW)
        negative: int = 5,
        workers: int = 4,
        random_state: int = 42
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs
        self.dm = dm
        self.negative = negative
        self.workers = workers
        self.random_state = random_state
        self.model: Doc2Vec = None

    def fit(self, texts: List[str], doc_tags: List[Union[int, str]] = None):
        """
        Builds vocabulary and trains Doc2Vec model on tagged documents.
        """
        if doc_tags is None:
            doc_tags = list(range(len(texts)))

        print(f"Preparing {len(texts)} TaggedDocuments for Doc2Vec training...")
        tagged_docs = []
        for tag, text in zip(doc_tags, texts):
            tokens = simple_tokenize(text)
            tagged_docs.append(TaggedDocument(words=tokens, tags=[str(tag)]))

        print(f"Initializing Doc2Vec model (vector_size={self.vector_size}, dm={self.dm}, epochs={self.epochs})...")
        self.model = Doc2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            epochs=self.epochs,
            dm=self.dm,
            negative=self.negative,
            workers=self.workers,
            seed=self.random_state
        )

        print("Building Doc2Vec vocabulary...")
        self.model.build_vocab(tagged_docs)
        print(f"Vocabulary size: {len(self.model.wv)} words.")

        print(f"Training Doc2Vec model for {self.epochs} epochs...")
        self.model.train(
            tagged_docs,
            total_examples=self.model.corpus_count,
            epochs=self.model.epochs
        )
        print("Doc2Vec training completed.")

    def get_document_vector(self, tag: Union[int, str]) -> np.ndarray:
        """
        Retrieves the learned vector for an existing tagged document.
        """
        tag_str = str(tag)
        if tag_str in self.model.dv:
            return self.model.dv[tag_str]
        else:
            raise KeyError(f"Document tag '{tag_str}' not found in Doc2Vec document vectors.")

    def infer_vector(self, text: str, epochs: int = 25) -> np.ndarray:
        """
        Infers an embedding vector for unseen legal text.
        """
        tokens = simple_tokenize(text)
        return self.model.infer_vector(tokens, epochs=epochs)

    def transform(self, texts: List[str], doc_tags: List[Union[int, str]] = None) -> np.ndarray:
        """
        Returns an array of document vectors (N x vector_size).
        If doc_tags are provided and already in the model, uses the trained dv,
        otherwise infers the vectors.
        """
        features = np.zeros((len(texts), self.vector_size), dtype=np.float32)
        for i, text in enumerate(texts):
            tag_str = str(doc_tags[i]) if doc_tags is not None else None
            if tag_str and tag_str in self.model.dv:
                features[i] = self.model.dv[tag_str]
            else:
                features[i] = self.infer_vector(text)
        return features

    def most_similar_to_tag(self, tag: Union[int, str], topn: int = 10) -> List[Tuple[str, float]]:
        """
        Finds the most similar documents to a given document tag using cosine similarity.
        """
        return self.model.dv.most_similar(str(tag), topn=topn)

    def similarity_between_tags(self, tag1: Union[int, str], tag2: Union[int, str]) -> float:
        """
        Computes cosine similarity between two document vectors.
        """
        return float(self.model.dv.similarity(str(tag1), str(tag2)))

    def save(self, output_dir: str = "outputs/models"):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "doc2vec_model.gensim")
        self.model.save(path)
        print(f"Saved Doc2Vec model to {path}")

    @classmethod
    def load(cls, output_dir: str = "outputs/models"):
        instance = cls()
        path = os.path.join(output_dir, "doc2vec_model.gensim")
        instance.model = Doc2Vec.load(path)
        instance.vector_size = instance.model.vector_size
        return instance
