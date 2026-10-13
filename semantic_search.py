import os
import json
import hashlib
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
import re

import nltk
import tiktoken
import numpy as np
import chromadb
from pypdf import PdfReader
from bs4 import BeautifulSoup
import docx

from openai_client import EmbeddingModel
from utils import get_logger


# ----------------------------- Setup Logging -----------------------------
semantic_logger = get_logger(name= "semantic_search", filename= "semantic_search.log", level= "debug")

class SemanticSearch:
    """
    Semantic Search for Retrieval-Augmented Generation (RAG) system.

    """

    FILE_INDEX = ".kb_index.json"
    CHUNK_INDEX = ".chunk_index.json"
    session_id = 0
    query_tokens = 0
    kb_tokens = 0
    RELEVANCY = 0.0
    ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".html", ".htm"}
    SYNC_INTERVAL = 600
    MAX_CONTEXT_CHARS = 12000

    def __init__(self, embeddings_model= "text-embedding-3-small", batch_size = 100,k_results=3, sid_prefix:str = ""):

        self.EMBEDDING_MODEL_NAME = embeddings_model
        self.kb_path = Path("knowledge_base")
        self.kb_path.mkdir(exist_ok=True)
        self.db_path = "data/vector_database"
        self.sid_prefix = sid_prefix
        self.embedding_model = EmbeddingModel(self.EMBEDDING_MODEL_NAME)
        self.batch_size = batch_size
        self.k_results = k_results

        

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)


        self.embedding_cache = OrderedDict()
        self.EMBEDDING_CACHE_MAX_SIZE = 10000

        self._last_sync_time = 0

        self.client = chromadb.PersistentClient(path=self.db_path)

        self.collection = self.client.get_or_create_collection(
            name="knowledge_vectors",
            metadata={"hnsw:space": "cosine"}
        )

        self.file_index_path = self.kb_path / self.FILE_INDEX
        self.chunk_index_path = self.kb_path / self.CHUNK_INDEX

        self.file_index = self._load_json(self.file_index_path)
        self.chunk_index = self._load_json(self.chunk_index_path)

        if self._needs_full_rebuild():
            self.client.delete_collection(name="knowledge_vectors")
            self.collection = self.client.get_or_create_collection(
                name="knowledge_vectors",
                metadata={"hnsw:space": "cosine"}
            )

            self.file_index = {}
            self.chunk_index = {}
            self._save_indexes()

        self.sync_knowledge_base(force=True)
    def _session_id(self):
        return self.sid_prefix +"-"+ str(self.session_id)
    # ----------------------------- INDEX -----------------------------
    def _load_json(self, path):
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _atomic_write(self, path, data):
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    def _save_indexes(self):
        self._atomic_write(self.file_index_path, self.file_index)
        self._atomic_write(self.chunk_index_path, self.chunk_index)

    # ----------------------------- HASH -----------------------------
    def _vector_id(self, file_id, chunk_index):
        return hashlib.sha256(f"{file_id}_{chunk_index}".encode()).hexdigest()

    def _file_hash(self, path):
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(8192):
                sha.update(chunk)
        return sha.hexdigest()

    def _chunk_hash(self, text):
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    # ----------------------------- EMBEDDING -----------------------------
    def _get_embedding_dimension(self):
        if hasattr(self, "_embedding_dimension"):
            return self._embedding_dimension
        vec = self.embedding_model.generate_embedding("dimension test: Tokens counted as kb_tokens")
        self.kb_tokens += self.embedding_model.tokens
        self._embedding_dimension = len(vec)
        return self._embedding_dimension

    def _embed_batches(self, chunks, chunk_hashes):

        embeddings = [None] * len(chunks)

        to_compute = []
        compute_indices = []

        # Check cache
        for i, (chunk, chash) in enumerate(
            zip(chunks, chunk_hashes)
        ):
            if chash in self.embedding_cache:
                embeddings[i] = self.embedding_cache[chash]
                self.embedding_cache.move_to_end(chash)
            else:
                to_compute.append(chunk)
                compute_indices.append(i)
        # Batch compute missing embeddings
        if to_compute:
            new_embs = []
            for start in range(
                0,
                len(to_compute),
                self.batch_size
            ):
                batch = to_compute[start:start + self.batch_size]
                batch_embs = (self.embedding_model.generate_batch_embeddings(batch))
                new_embs.extend(batch_embs)
                self.kb_tokens += self.embedding_model.batch_tokens 

                

            # Store results
            for idx, emb in zip(compute_indices, new_embs):
                chash = chunk_hashes[idx]
                embeddings[idx] = emb
                self.embedding_cache[chash] = emb
                self.embedding_cache.move_to_end(chash)
                
                # LRU eviction
                if (
                    len(self.embedding_cache)
                    > self.EMBEDDING_CACHE_MAX_SIZE
                ):
                    self.embedding_cache.popitem(last=False)

        return embeddings

    # ----------------------------- EXTRACTION -----------------------------
    def _clean_pdf_text(self,text: str) -> str:
    # remove excessive whitespace
        text = " ".join(text.split())

        # remove common header/footer artifacts (optional patterns)
        text = re.sub(r"Page\s*\d+\s*of\s*\d+", "", text, flags=re.IGNORECASE)

        return text.strip()


    def _extract_pdf(self, path):
        reader = PdfReader(path)

        pages = []
        for page in reader.pages:
            raw = page.extract_text() or ""
            cleaned = self._clean_pdf_text(raw)

            if cleaned:
                pages.append(cleaned)

        return "\n".join(pages)

    def _extract_docx(self, path):
        doc = docx.Document(path)
        return "\n".join(p.text for p in doc.paragraphs)

    def _extract_html(self, path):
        with open(path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
        return soup.get_text(separator="\n")

    def _extract_text(self, path):
        ext = path.suffix.lower()
        try:
            if ext == ".pdf":
                return self._extract_pdf(path)
            elif ext == ".docx":
                return self._extract_docx(path)
            elif ext in [".html", ".htm"]:
                return self._extract_html(path)
            elif ext in [".txt", ".md"]:
                return path.read_text(encoding="utf-8")
        except Exception as e:
            semantic_logger.error(f"Error extracting {path}: {e}")
        return ""

    # ----------------------------- CHUNKING -----------------------------
    def _estimate_tokens(self, text: str) -> int:
        """
        Accurate token estimation using OpenAI tokenizer.
        """
        if not text:
            return 0
        try:
            enc = tiktoken.get_encoding("o200k_base")
            return len(enc.encode(text))
        except Exception:
            # fallback to heuristic
            return max(1, len(text) // 4)

    def _chunk_text(self, text, max_tokens=512, overlap_sentences=1):
        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        # Sentence splitting
        try:
            sentences = nltk.sent_tokenize(text)
        except Exception:
            sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current = []
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_tokens = self._estimate_tokens(sentence)

            # ------------------------------------------------
            # Normal chunking
            # ------------------------------------------------
            if (current_tokens + sentence_tokens > max_tokens):
                if current:
                    chunks.append(" ".join(current))
                overlap = (current[-overlap_sentences:] if current else [])
                current = overlap + [sentence]
                current_tokens = sum(self._estimate_tokens(s) for s in current)
            else:
                current.append(sentence)
                current_tokens += sentence_tokens
        # Final chunk
        if current:
            chunks.append(" ".join(current))
        return chunks

    # ----------------------------- REBUILD -----------------------------
    def _needs_full_rebuild(self):
        data = self.collection.get(limit=1, include=["metadatas", "embeddings"])
        metas = data.get("metadatas")
        embeddings = data.get("embeddings")

        if metas is None or len(metas) == 0:
            return False

        stored_model = metas[0].get("embedding_model")
        if stored_model != self.EMBEDDING_MODEL_NAME:
            semantic_logger.info(f"Rebuild triggered: The embedding model was changed from {stored_model} to {self.EMBEDDING_MODEL_NAME}") 
            return True

        if embeddings is None:
            return False

        if isinstance(embeddings, np.ndarray):
            if embeddings.size == 0:
                return False
            first_embedding = embeddings[0]
        else:
            if len(embeddings) == 0:
                return False
            first_embedding = embeddings[0]

        while isinstance(first_embedding, (list, np.ndarray)) and len(first_embedding) > 0:
            if isinstance(first_embedding[0], (list, np.ndarray)):
                first_embedding = first_embedding[0]
            else:
                break

        if first_embedding is None:
            return False

        stored_dim = len(first_embedding)
        if stored_dim != self._get_embedding_dimension():
            semantic_logger.info(f"Rebuild triggered: The embedding dimension mismatch {stored_dim} vs {self._get_embedding_dimension()}")             
            return True
        
        else: 
            return False 

    # ----------------------------- PROCESS -----------------------------
    def _process_file(self, filename):
        path = self.kb_path / filename

        try:
            file_id = str(path.relative_to(self.kb_path)).replace("\\", "/")
            file_hash = self._file_hash(path)

            if file_id in self.file_index and self.file_index[file_id] == file_hash:
                return None

            text = self._extract_text(path)
            if not text.strip():
                return None

            chunks = self._chunk_text(text)
            chunk_hashes = [self._chunk_hash(c) for c in chunks]

            return file_id, chunks, chunk_hashes, file_hash

        except Exception as e:
            semantic_logger.error(f"Failed processing at Semantic Search._process_file for {filename}: {e}")
            return None

    # ----------------------------- SYNC -----------------------------
    def sync_knowledge_base(self, force=False):

        if not force and (time.time() - self._last_sync_time < self.SYNC_INTERVAL):
            return
        
        semantic_logger.info(f"Session ID: {self._session_id()} Syncing knowledge-base and vector database triggered. Forced: {force}")
        self._last_sync_time = time.time()
        start = time.time()
        files = [
            f for f in os.listdir(self.kb_path)
            if (self.kb_path / f).is_file()
            and Path(f).suffix.lower() in self.ALLOWED_EXTENSIONS
        ]

        current_files = {
            str((self.kb_path / f).relative_to(self.kb_path)).replace("\\", "/")
            for f in files
        }

        deleted = set(self.file_index.keys()) - current_files

        for file_id in deleted:
            for idx in self.chunk_index.get(file_id, {}).keys():
                self.collection.delete(ids=[self._vector_id(file_id, int(idx))])

            self.chunk_index.pop(file_id, None)
            self.file_index.pop(file_id, None)

        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(self._process_file, files))

        for result in results:
            if not result:
                continue

            file_id, chunks, chunk_hashes, file_hash = result
            for retry in range(3):
                try:
                    self._update_file(file_id, chunks, chunk_hashes, file_hash)
                    self.file_index[file_id] = file_hash
                    break
                except Exception as e:
                    semantic_logger.error(f"_update_file failed in sync_knowledge_base: Retrying {retry+1}/3")
                    if retry == 2:
                        semantic_logger.critical(f"_update_file in sync_knowledge_base FAILED- KB integrity compromised")



        self._save_indexes()
        end = time.time()
        semantic_logger.info(f"Session ID: {self._session_id()} Successfully synced knowleedge-base in {end - start:.4f} seconds. . . ")

    # ----------------------------- UPDATE -----------------------------
    def _update_file(self, file_id, chunks, chunk_hashes, file_hash):
        stored_chunks = self.chunk_index.get(file_id, {})

        embeddings_needed, embed_indexes, embed_hashes = [], [], []

        for i, chash in enumerate(chunk_hashes):
            if stored_chunks.get(str(i)) != chash:
                embeddings_needed.append(chunks[i])
                embed_indexes.append(i)
                embed_hashes.append(chash)

        if embeddings_needed:
            try:
                embeddings = self._embed_batches(embeddings_needed, embed_hashes)
            except Exception as e:
                semantic_logger.error(f"SemanticSearch._update_file: Embedding failed for file {file_id}: {e}")
                raise e

            for idx, emb, chash in zip(embed_indexes, embeddings, embed_hashes):
                self.collection.upsert(
                    ids=[self._vector_id(file_id, idx)],
                    documents=[chunks[idx]],
                    embeddings=[emb],
                    metadatas=[{
                        "filename": file_id,
                        "chunk_index": idx,
                        "filehash": file_hash,
                        "chunkhash": chash,
                        "embedding_model": self.EMBEDDING_MODEL_NAME
                    }]
                )

        stored_indexes = set(map(int, stored_chunks.keys()))
        current_indexes = set(range(len(chunks)))

        for idx in stored_indexes - current_indexes:
            self.collection.delete(ids=[self._vector_id(file_id, idx)])

        self.chunk_index[file_id] = {
            str(i): chunk_hashes[i] for i in range(len(chunks))
        }

    # ----------------------------- QUERY -----------------------------
    def query(self, text:str, filename:str=None)-> dict:
        self.session_id += 1
        self.RELEVANCY = 0.0
        self.query_tokens = 0
        
        self.sync_knowledge_base()


        # ---------------------------------------------------------
        # Check the vector database for embeddings
        # ---------------------------------------------------------
        if self.collection.count() == 0:
            semantic_logger.error(f"Session ID: {self._session_id()} Vector Database is empty. Check knowledge_base directory for files and re-run program") 
            return {}
        
        # ---------------------------------------------------------
        # generate embeddings for querying
        # ---------------------------------------------------------        
        query_vector = self.embedding_model.generate_embedding(text=text)
        self.query_tokens = self.embedding_model.tokens 
        
        where_filter = {"filename": filename} if filename else None
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=self.k_results,
            where=where_filter,
            include=["documents", "distances", "metadatas"])
        docs = results.get("documents", [[]])[0] or []
        distances = results.get("distances", [[]])[0] or []
        metas = results.get("metadatas", [[]])[0] or []
        payload = {}
        total_chars = 0
        count = 1
        for doc, dist, meta in zip(docs, distances, metas):
            if not doc or meta is None:
                continue
            raw = str(doc)
            clean = " ".join(raw.split())
            relevancy = 1 - dist if dist is not None else 0

            entry = {"Filename": meta.get('filename'),
                    "Relevancy Score": f"{relevancy:.4f}",
                    "Context": clean}
            if total_chars + len(clean) > self.MAX_CONTEXT_CHARS:
                break
            payload[f"{count}"] = entry
            self.RELEVANCY += relevancy
            total_chars += len(clean)
            count += 1 
        
        return payload