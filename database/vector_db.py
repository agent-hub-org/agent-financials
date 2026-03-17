import os
from typing import List, Dict, Any

from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 200
_EXIST_THRESHOLD = 0.7
_UPSERT_BATCH = 100


class VectorDB:
    # Embedding dimensions per provider
    _DIMENSIONS = {"gemini": 3072, "nvidia": 4096}

    def __init__(self, provider: str = "nvidia"):
        self.provider = provider
        self.pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = "financial-reports"

        if provider == "gemini":
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="gemini-embedding-2-preview",
                google_api_key=os.getenv("GEMINI_API_KEY"),
            )
        else:
            self.embeddings = NVIDIAEmbeddings(
                model="nvidia/nv-embed-v1",
                nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
            )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=_CHUNK_SIZE,
            chunk_overlap=_CHUNK_OVERLAP,
        )
        self._ensure_index()

    def _ensure_index(self):
        expected_dim = self._DIMENSIONS[self.provider]
        existing = {idx.name: idx for idx in self.pinecone.list_indexes()}

        if self.index_name in existing:
            current_dim = existing[self.index_name].dimension
            if current_dim != expected_dim:
                self.pinecone.delete_index(self.index_name)
                existing.pop(self.index_name)

        if self.index_name not in existing:
            self.pinecone.create_index(
                name=self.index_name,
                dimension=expected_dim,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

        self.index = self.pinecone.Index(self.index_name)

    def upsert_reports(self, ticker: str, reports_data: List[Dict[str, Any]]):
        """
        Chunk, embed, and upsert financial reports into Pinecone.
        Each dictionary should have:
        - title: str
        - content: str (the markdown or text representation of the report)
        - period: str (e.g., "2023-Q1")
        - type: str (e.g., "Income Statement")
        """
        upsert_data = []

        for report in reports_data:
            content = report["content"]
            chunks = self.splitter.split_text(content)
            vectors = self.embeddings.embed_documents(chunks)

            # Create a unique document ID base
            doc_id = f"{ticker}_{report['period']}_{report['type'].replace(' ', '_')}"

            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                upsert_data.append({
                    "id": f"{doc_id}_chunk_{i}",
                    "values": vector,
                    "metadata": {
                        "ticker": ticker,
                        "title": report["title"],
                        "period": report["period"],
                        "type": report["type"],
                        "chunk_index": i,
                        "text": chunk,
                    },
                })

        for i in range(0, len(upsert_data), _UPSERT_BATCH):
            self.index.upsert(vectors=upsert_data[i:i + _UPSERT_BATCH])

    def reports_exist(self, ticker: str) -> bool:
        """Check if documents for a ticker exist in the vector DB."""
        # Querying with a dummy vector but filtering by ticker
        dummy_vector = [0.0] * self._DIMENSIONS[self.provider]
        results = self.index.query(
            vector=dummy_vector,
            top_k=1,
            filter={"ticker": {"$eq": ticker}},
            include_metadata=False,
        )
        return bool(results.matches)

    def retrieve(self, query: str, ticker: str | None = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Semantically search the vector DB and return the top_k matching chunks."""
        query_vector = self.embeddings.embed_query(query)
        
        filter_dict = {}
        if ticker:
            filter_dict["ticker"] = {"$eq": ticker}

        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            filter=filter_dict if filter_dict else None,
            include_metadata=True,
        )

        chunks = []
        for match in results.matches:
            meta = match.metadata
            if meta:
                chunks.append({
                    "ticker": meta.get("ticker"),
                    "title": meta.get("title"),
                    "period": meta.get("period"),
                    "type": meta.get("type"),
                    "text": meta.get("text"),
                    "score": match.score,
                })
        return chunks
