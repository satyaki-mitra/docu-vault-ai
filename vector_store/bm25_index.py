"""
BM25 Index
Keyword-based search using BM25 algorithm
"""

from typing import List, Tuple, Optional
from pathlib import Path
import pickle
from rank_bm25 import BM25Okapi
import numpy as np

from config.logging_config import get_logger
from config.settings import get_settings
from config.models import DocumentChunk

logger = get_logger(__name__)
settings = get_settings()


class BM25Index:
    """
    BM25 (Best Matching 25) keyword search index.
    Provides lexical search to complement semantic vector search.
    """
    
    def __init__(
        self,
        k1: float = None,
        b: float = None
    ):
        """
        Initialize BM25 index.
        
        Args:
            k1: Term frequency saturation parameter (default from settings)
            b: Length normalization parameter (default from settings)
        """
        self.k1 = k1 if k1 is not None else settings.BM25_K1
        self.b = b if b is not None else settings.BM25_B
        
        self.index: Optional[BM25Okapi] = None
        self.corpus: List[List[str]] = []  # Tokenized documents
        self.chunk_ids: List[str] = []
        self.raw_texts: List[str] = []  # Original texts for reference
        
        self.logger = logger
        self.logger.info(f"BM25Index initialized: k1={self.k1}, b={self.b}")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Args:
            text: Input text
        
        Returns:
            List of tokens
        """
        # Simple whitespace tokenization with lowercasing
        # For production, consider more sophisticated tokenization
        tokens = text.lower().split()
        
        # Remove very short tokens
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens
    
    def add_documents(
        self,
        texts: List[str],
        chunk_ids: List[str]
    ):
        """
        Add documents to BM25 index.
        
        Args:
            texts: List of document texts
            chunk_ids: List of chunk IDs
        """
        if len(texts) != len(chunk_ids):
            raise ValueError("Number of texts must match number of chunk IDs")
        
        # Tokenize documents
        for text in texts:
            tokens = self._tokenize(text)
            self.corpus.append(tokens)
        
        self.chunk_ids.extend(chunk_ids)
        self.raw_texts.extend(texts)
        
        # Rebuild index
        if self.corpus:
            self.index = BM25Okapi(self.corpus, k1=self.k1, b=self.b)
        
        self.logger.info(
            f"Added {len(texts)} documents (total: {len(self.corpus)})"
        )
    
    def search(
        self,
        query: str,
        k: int = 10
    ) -> Tuple[List[str], List[float]]:
        """
        Search using BM25.
        
        Args:
            query: Search query
            k: Number of results to return
        
        Returns:
            Tuple of (chunk_ids, scores)
        """
        if self.index is None or not self.corpus:
            self.logger.warning("Index is empty")
            return [], []
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        if not query_tokens:
            self.logger.warning("Query resulted in no tokens")
            return [], []
        
        # Get BM25 scores
        scores = self.index.get_scores(query_tokens)
        
        # Get top k indices
        k = min(k, len(scores))
        top_indices = np.argsort(scores)[::-1][:k]
        
        # Get corresponding chunk IDs and scores
        chunk_ids = [self.chunk_ids[idx] for idx in top_indices]
        top_scores = [float(scores[idx]) for idx in top_indices]
        
        return chunk_ids, top_scores
    
    def batch_search(
        self,
        queries: List[str],
        k: int = 10
    ) -> Tuple[List[List[str]], List[List[float]]]:
        """
        Search for multiple queries.
        
        Args:
            queries: List of queries
            k: Number of results per query
        
        Returns:
            Tuple of (chunk_ids_list, scores_list)
        """
        all_chunk_ids = []
        all_scores = []
        
        for query in queries:
            chunk_ids, scores = self.search(query, k=k)
            all_chunk_ids.append(chunk_ids)
            all_scores.append(scores)
        
        return all_chunk_ids, all_scores
    
    def get_document(self, chunk_id: str) -> Optional[str]:
        """
        Get original document text by chunk ID.
        
        Args:
            chunk_id: Chunk ID
        
        Returns:
            Original text or None
        """
        try:
            idx = self.chunk_ids.index(chunk_id)
            return self.raw_texts[idx]
        except ValueError:
            return None
    
    def get_statistics(self) -> dict:
        """
        Get index statistics.
        
        Returns:
            Statistics dictionary
        """
        if not self.corpus:
            return {
                "num_documents": 0,
                "avg_doc_length": 0,
                "total_tokens": 0,
                "k1": self.k1,
                "b": self.b,
            }
        
        doc_lengths = [len(doc) for doc in self.corpus]
        total_tokens = sum(doc_lengths)
        
        return {
            "num_documents": len(self.corpus),
            "avg_doc_length": np.mean(doc_lengths),
            "min_doc_length": min(doc_lengths),
            "max_doc_length": max(doc_lengths),
            "total_tokens": total_tokens,
            "k1": self.k1,
            "b": self.b,
        }
    
    def save(self, directory: Path):
        """
        Save BM25 index to disk.
        
        Args:
            directory: Directory to save files
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save all data
        data = {
            "corpus": self.corpus,
            "chunk_ids": self.chunk_ids,
            "raw_texts": self.raw_texts,
            "k1": self.k1,
            "b": self.b,
        }
        
        save_path = directory / "bm25_index.pkl"
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        
        self.logger.info(f"Saved BM25 index to {directory}")
    
    def load(self, directory: Path):
        """
        Load BM25 index from disk.
        
        Args:
            directory: Directory containing saved files
        """
        directory = Path(directory)
        load_path = directory / "bm25_index.pkl"
        
        if not load_path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {load_path}")
        
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        
        self.corpus = data["corpus"]
        self.chunk_ids = data["chunk_ids"]
        self.raw_texts = data["raw_texts"]
        self.k1 = data["k1"]
        self.b = data["b"]
        
        # Rebuild BM25 index
        if self.corpus:
            self.index = BM25Okapi(self.corpus, k1=self.k1, b=self.b)
        
        self.logger.info(
            f"Loaded BM25 index from {directory} ({len(self.corpus)} documents)"
        )
    
    def reset(self):
        """Reset index"""
        self.index = None
        self.corpus = []
        self.chunk_ids = []
        self.raw_texts = []
        self.logger.info("Index reset")
    
    def get_term_frequencies(self, query: str) -> dict:
        """
        Get term frequencies for query across corpus.
        
        Args:
            query: Search query
        
        Returns:
            Dictionary of term frequencies
        """
        if not self.corpus:
            return {}
        
        query_tokens = self._tokenize(query)
        term_freqs = {}
        
        for token in query_tokens:
            # Count documents containing this term
            count = sum(1 for doc in self.corpus if token in doc)
            term_freqs[token] = {
                "doc_frequency": count,
                "doc_percentage": (count / len(self.corpus)) * 100
            }
        
        return term_freqs
    
    def explain_score(self, query: str, chunk_id: str) -> dict:
        """
        Explain BM25 score for a specific document.
        
        Args:
            query: Search query
            chunk_id: Chunk ID to explain
        
        Returns:
            Dictionary with score breakdown
        """
        try:
            idx = self.chunk_ids.index(chunk_id)
        except ValueError:
            return {"error": "Chunk ID not found"}
        
        query_tokens = self._tokenize(query)
        document = self.corpus[idx]
        
        if self.index is None:
            return {"error": "Index not initialized"}
        
        # Get BM25 score
        scores = self.index.get_scores(query_tokens)
        score = float(scores[idx])
        
        # Get term statistics
        term_info = []
        for token in query_tokens:
            term_freq_in_doc = document.count(token)
            term_info.append({
                "term": token,
                "frequency_in_doc": term_freq_in_doc,
                "in_document": token in document
            })
        
        return {
            "chunk_id": chunk_id,
            "total_score": score,
            "doc_length": len(document),
            "query_terms": term_info,
            "avg_doc_length": np.mean([len(d) for d in self.corpus])
        }


if __name__ == "__main__":
    # Test BM25 index
    print("=== BM25 Index Tests ===\n")
    
    # Test 1: Create and add documents
    print("Test 1: Create and add documents")
    bm25 = BM25Index()
    
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Natural language processing enables computers to understand text",
        "Computer vision allows machines to interpret images",
        "Artificial intelligence is transforming technology"
    ]
    chunk_ids = [f"chunk_{i}" for i in range(len(texts))]
    
    bm25.add_documents(texts, chunk_ids)
    print(f"  Added {len(texts)} documents")
    
    stats = bm25.get_statistics()
    print(f"  Stats: {stats['num_documents']} docs, "
          f"avg length: {stats['avg_doc_length']:.1f} tokens")
    print()
    
    # Test 2: Search
    print("Test 2: Search")
    query = "artificial intelligence machine learning"
    results, scores = bm25.search(query, k=3)
    
    print(f"  Query: '{query}'")
    for i, (chunk_id, score) in enumerate(zip(results, scores)):
        text = bm25.get_document(chunk_id)
        print(f"    {i+1}. {chunk_id} (score: {score:.4f})")
        print(f"       {text}")
    print()
    
    # Test 3: Term frequencies
    print("Test 3: Term frequencies")
    term_freqs = bm25.get_term_frequencies(query)
    print(f"  Query: '{query}'")
    for term, freq in term_freqs.items():
        print(f"    '{term}': {freq['doc_frequency']} docs ({freq['doc_percentage']:.1f}%)")
    print()
    
    # Test 4: Score explanation
    print("Test 4: Score explanation")
    if results:
        explanation = bm25.explain_score(query, results[0])
        print(f"  Explaining score for: {results[0]}")
        print(f"  Total score: {explanation['total_score']:.4f}")
        print(f"  Doc length: {explanation['doc_length']} tokens")
        print(f"  Query terms:")
        for term_info in explanation['query_terms']:
            print(f"    - {term_info['term']}: "
                  f"freq={term_info['frequency_in_doc']}, "
                  f"in_doc={term_info['in_document']}")
    print()
    
    # Test 5: Batch search
    print("Test 5: Batch search")
    queries = [
        "neural networks",
        "computer vision",
        "artificial intelligence"
    ]
    batch_results, batch_scores = bm25.batch_search(queries, k=2)
    
    for query, results, scores in zip(queries, batch_results, batch_scores):
        print(f"  Query: '{query}' -> {len(results)} results")
    print()
    
    # Test 6: Save and load
    print("Test 6: Save and load")
    test_dir = Path("test_bm25_index")
    bm25.save(test_dir)
    print(f"  Saved to {test_dir}")
    
    bm25_2 = BM25Index()
    bm25_2.load(test_dir)
    stats2 = bm25_2.get_statistics()
    print(f"  Loaded: {stats2['num_documents']} documents")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    print()
    
    print("✓ BM25 index module created successfully!")