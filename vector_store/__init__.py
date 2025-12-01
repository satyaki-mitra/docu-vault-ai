# vector_store/__init__.py

"""
Vector Store Module
FAISS vector index, BM25 keyword index, and metadata storage
"""

# Import the core classes from the submodules
from vector_store.faiss_manager import FAISSManager
from vector_store.bm25_index import BM25Index
from vector_store.metadata_store import MetadataStore
from vector_store.index_builder import IndexBuilder # Import IndexBuilder

# --- Potentially needed imports if vector_search.py uses them directly ---
# from vector_store.faiss_manager import FAISSManager
# from vector_store.bm25_index import BM25Index
# from vector_store.metadata_store import MetadataStore

__all__ = [
    # Core Stores/Indices
    "FAISSManager",
    "BM25Index",
    "MetadataStore",
    # Index Builder
    "IndexBuilder",
    # Potentially add FAISSManager, BM25Index, MetadataStore if used directly elsewhere
]

# --- Global instance for shared IndexBuilder usage (if applicable) ---
# This is a common pattern to avoid re-initializing the builder and its components multiple times.
# The vector_search.py might be expecting a function like get_index_builder() that returns this instance.
_index_builder_instance = None

def get_index_builder() -> IndexBuilder:
    """
    Gets a shared instance of the IndexBuilder.
    This function is likely expected by modules like retrieval/vector_search.py.
    It ensures the builder and its underlying stores (FAISS, BM25, Metadata) are
    initialized only once and reused efficiently.

    Returns:
        IndexBuilder: A configured instance of the index builder.
    """
    global _index_builder_instance
    if _index_builder_instance is None:
        # Initialize the IndexBuilder with default settings
        # It might rely on settings from config/settings.py which are loaded implicitly
        # or passed during the instantiation of dependent classes.
        # For now, we'll initialize with defaults, assuming settings are handled by IndexBuilder itself.
        _index_builder_instance = IndexBuilder()
    return _index_builder_instance

# Add get_index_builder to __all__ if it's intended to be a public API
__all__.append("get_index_builder")

# --- Example of how other functions might be exposed if needed ---
# def get_faiss_manager() -> FAISSManager:
#     # Similar singleton or factory pattern for FAISSManager if needed elsewhere
#     pass