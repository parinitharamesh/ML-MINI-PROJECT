"""
Transformer / BERT Pipeline for Long Legal Texts.
Directly addresses Experiment 3 / Future Work in Iyer (2020).
Solves the Out-Of-Memory (OOM) bottleneck on legal opinions using
Head+Tail chunk pooling and contextual representation extraction.
"""

import os
import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from transformers import BertTokenizer, BertModel


class LegalTransformerExtractor:
    def __init__(
        self,
        model_name: str = "prajjwal1/bert-tiny",
        max_tokens: int = 512,
        head_tokens: int = 256,
        tail_tokens: int = 256,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.head_tokens = head_tokens
        self.tail_tokens = tail_tokens
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading Transformer tokenizer & model ({self.model_name}) on device: {self.device}...")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name).to(self.device)
        self.model.eval()
        self.embedding_dim = self.model.config.hidden_size
        print(f"Transformer loaded. Hidden size: {self.embedding_dim} dim.")

    def chunk_head_tail(self, text: str) -> str:
        """
        Extracts legal reasoning by combining the introductory syllabus/facts (head)
        with the holding/conclusion (tail), avoiding the OOM quadratic explosion.
        """
        words = text.split()
        if len(words) <= self.max_tokens:
            return text
        
        # Take head words and tail words
        head_words = words[:self.head_tokens]
        tail_words = words[-self.tail_tokens:]
        return " ".join(head_words + tail_words)

    @torch.no_grad()
    def extract_document_embedding(self, text: str) -> np.ndarray:
        """
        Generates a fixed-size contextual embedding for a legal document
        via mean-pooling over the last hidden state of active tokens.
        """
        chunked_text = self.chunk_head_tail(text)
        inputs = self.tokenizer(
            chunked_text,
            max_length=self.max_tokens,
            truncation=True,
            padding=False,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)
        # outputs.last_hidden_state: [1, seq_len, hidden_size]
        last_hidden = outputs.last_hidden_state
        attention_mask = inputs['attention_mask'].unsqueeze(-1) # [1, seq_len, 1]

        # Mean pooling over non-padded tokens
        sum_embeddings = torch.sum(last_hidden * attention_mask, dim=1)
        sum_mask = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
        mean_pooled = (sum_embeddings / sum_mask).squeeze(0).cpu().numpy()
        return mean_pooled

    def extract_batch_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Extracts contextual embeddings for a collection of legal texts.
        """
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                vec = self.extract_document_embedding(text)
                embeddings.append(vec)
            if (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(texts):
                print(f"Extracted Transformer embeddings: {min(i + batch_size, len(texts))}/{len(texts)}")
        return np.vstack(embeddings)
