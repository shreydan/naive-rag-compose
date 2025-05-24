import faiss
import pickle
import os
import numpy as np
from typing import List, Union, Tuple
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

class FAISSHandler:
    def __init__(
        self,
        store_path: Path,
    ):
        self.store_path = store_path
        self._load()
    
    def mean_pooling(self, model_output, attention_mask) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def create_embeddings(self, texts:List[str], prefix:str='clustering') -> List[np.ndarray]:
        encoding_texts = [f"{prefix}: {t}" for t in texts]
        print(encoding_texts)
        encoded_input = self.tokenizer(encoding_texts, padding='longest', truncation=True, return_tensors='pt')

        matryoshka_dim = 512

        with torch.no_grad():
            model_output = self.model(**encoded_input)
        print(model_output['last_hidden_state'].shape)
        embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
        embeddings = embeddings[:, :matryoshka_dim]
        embeddings = F.normalize(embeddings, p=2, dim=1)
        text_embeds = [t.unsqueeze(0).numpy() for t in embeddings.unbind(0)]
        return text_embeds

    def add_documents(self, texts: List[str]):
        embeddings = self.create_embeddings(texts)
        for embedding in embeddings:
            self.index.add(embedding)
        self.documents.extend(texts)

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_embedding = self.create_embeddings([query],prefix='search_query')[0]
        _, indices = self.index.search(query_embedding, top_k)

        results = [self.documents[idx] for idx in indices[0] if idx < len(self.documents)]
        return results

    def _save(self):
        faiss_bytes = faiss.serialize_index(self.index)
        with open(self.store_path, "wb") as f:
            pickle.dump({"index": faiss_bytes, "documents": self.documents}, f)

    def _load(self):
        if self.store_path.exists():
            with open(self.store_path, "rb") as f:
                data = pickle.load(f)
                self.index = faiss.deserialize_index(data["index"])
                self.documents = data["documents"]
        else:
            self.index = faiss.IndexFlatL2(512)
            self.documents = []

        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', local_files_only=True, cache_dir='/app/.cache')
        self.model = AutoModel.from_pretrained('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True, safe_serialization=True, local_files_only=True, cache_dir='/app/.cache')
        self.model.eval()

    def _reset(self):
        if self.store_path.exists():
            self.index = faiss.IndexFlatL2(512)
            self.documents = []
            self._save()
