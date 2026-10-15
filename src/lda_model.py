"""
Latent Dirichlet Allocation (LDA) Topic Model with TF-IDF Representation.
Corresponds to Experiment 1 in Iyer (2020).
Extracts latent topic distributions from legal text using Gensim.
"""

import os
import joblib
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaModel
from src.data_loader import simple_tokenize


class LegalLDAPipeline:
    def __init__(
        self,
        num_topics: int = 50,
        no_below: int = 15,
        no_above: float = 0.5,
        passes: int = 10,
        random_state: int = 42
    ):
        self.num_topics = num_topics
        self.no_below = no_below
        self.no_above = no_above
        self.passes = passes
        self.random_state = random_state
        self.dictionary: Dictionary = None
        self.tfidf_model: TfidfModel = None
        self.lda_model: LdaModel = None

    def fit(self, texts: List[str]):
        """
        Tokenizes texts, builds dictionary, computes TF-IDF, and fits Gensim LDA model.
        """
        print(f"Tokenizing {len(texts)} documents for LDA...")
        tokenized_docs = [simple_tokenize(text) for text in texts]

        print("Building Gensim Dictionary...")
        self.dictionary = Dictionary(tokenized_docs)
        original_vocab_len = len(self.dictionary)
        self.dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above)
        print(f"Filtered dictionary: {original_vocab_len} -> {len(self.dictionary)} vocabulary terms.")

        bow_corpus = [self.dictionary.doc2bow(doc) for doc in tokenized_docs]

        print("Fitting TF-IDF model...")
        self.tfidf_model = TfidfModel(bow_corpus)
        tfidf_corpus = self.tfidf_model[bow_corpus]

        print(f"Training LDA model with {self.num_topics} topics (passes={self.passes})...")
        self.lda_model = LdaModel(
            corpus=tfidf_corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=self.random_state,
            passes=self.passes,
            alpha='auto',
            eta='auto',
            eval_every=None
        )
        print("LDA training completed.")

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transforms texts into dense document-topic probability distribution vectors (N x num_topics).
        """
        if self.lda_model is None or self.dictionary is None:
            raise ValueError("Model must be fitted before transforming texts.")

        features = np.zeros((len(texts), self.num_topics), dtype=np.float32)
        for i, text in enumerate(texts):
            tokens = simple_tokenize(text)
            bow = self.dictionary.doc2bow(tokens)
            tfidf_doc = self.tfidf_model[bow]
            topic_dist = self.lda_model.get_document_topics(tfidf_doc, minimum_probability=0.0)
            for topic_id, prob in topic_dist:
                features[i, topic_id] = prob

        return features

    def get_top_topic_words(self, topn: int = 7) -> Dict[int, List[Tuple[str, float]]]:
        """
        Returns the top words for each discovered topic.
        """
        topics = {}
        for topic_id in range(self.num_topics):
            topics[topic_id] = self.lda_model.show_topic(topic_id, topn=topn)
        return topics

    def save(self, output_dir: str = "outputs/models"):
        os.makedirs(output_dir, exist_ok=True)
        self.lda_model.save(os.path.join(output_dir, "lda_model.gensim"))
        self.dictionary.save(os.path.join(output_dir, "lda_dict.gensim"))
        joblib.dump(self.tfidf_model, os.path.join(output_dir, "tfidf_model.joblib"))
        print(f"Saved LDA artifacts to {output_dir}")

    @classmethod
    def load(cls, output_dir: str = "outputs/models"):
        instance = cls()
        instance.lda_model = LdaModel.load(os.path.join(output_dir, "lda_model.gensim"))
        instance.dictionary = Dictionary.load(os.path.join(output_dir, "lda_dict.gensim"))
        instance.tfidf_model = joblib.load(os.path.join(output_dir, "tfidf_model.joblib"))
        instance.num_topics = instance.lda_model.num_topics
        return instance
