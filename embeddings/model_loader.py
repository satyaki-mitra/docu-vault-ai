"""
Embedding Model Loader
Manages loading and caching of embedding models
"""

from typing import Optional, Dict
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer

from config.logging_config import get_logger
from config.settings import get_settings

logger = get_logger(__name__)
settings = get_settings()


class ModelLoader:
    """
    Singleton loader for embedding models.
    Handles model caching, device management, and optimization.
    """
    
    _instance = None
    _models: Dict[str, SentenceTransformer] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.logger = logger
        self.device = self._detect_device()
        self._initialized = True
        
        self.logger.info(f"ModelLoader initialized with device: {self.device}")
    
    def _detect_device(self) -> str:
        """
        Detect best available device for inference.
        
        Returns:
            Device string (cuda, mps, or cpu)
        """
        # Check settings preference
        preferred_device = settings.EMBEDDING_DEVICE
        
        if preferred_device == "cuda" and torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            self.logger.info(f"Using CUDA GPU: {gpu_name}")
        
        elif preferred_device == "mps" and torch.backends.mps.is_available():
            device = "mps"
            self.logger.info("Using Apple Metal Performance Shaders (MPS)")
        
        else:
            device = "cpu"
            self.logger.info("Using CPU for inference")
            if preferred_device != "cpu":
                self.logger.warning(
                    f"Requested device '{preferred_device}' not available, falling back to CPU"
                )
        
        return device
    
    def load_model(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
        cache_folder: Optional[Path] = None
    ) -> SentenceTransformer:
        """
        Load embedding model with caching.
        
        Args:
            model_name: HuggingFace model name (default from settings)
            device: Device to load model on (default: auto-detect)
            normalize_embeddings: Normalize embeddings to unit length
            cache_folder: Custom cache folder for model weights
        
        Returns:
            Loaded SentenceTransformer model
        """
        model_name = model_name or settings.EMBEDDING_MODEL
        device = device or self.device
        
        # Check if model already loaded
        cache_key = f"{model_name}_{device}"
        if cache_key in self._models:
            self.logger.debug(f"Using cached model: {model_name}")
            return self._models[cache_key]
        
        self.logger.info(f"Loading embedding model: {model_name}")
        
        try:
            # Load model
            model = SentenceTransformer(
                model_name,
                device=device,
                cache_folder=str(cache_folder) if cache_folder else None
            )
            
            # Configure model
            if normalize_embeddings:
                model.normalize_embeddings = True
            
            # Optimize for inference
            model.eval()  # Set to evaluation mode
            
            # Cache model
            self._models[cache_key] = model
            
            # Log model info
            self._log_model_info(model, model_name)
            
            return model
        
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _log_model_info(self, model: SentenceTransformer, model_name: str):
        """Log information about loaded model"""
        try:
            # Get embedding dimension
            dim = model.get_sentence_embedding_dimension()
            
            # Get model size (approximate)
            total_params = sum(
                p.numel() for p in model.parameters()
            )
            size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
            
            self.logger.info(
                f"Model loaded successfully: {model_name}\n"
                f"  Embedding dimension: {dim}\n"
                f"  Parameters: {total_params:,}\n"
                f"  Approximate size: {size_mb:.1f} MB\n"
                f"  Device: {self.device}"
            )
        except Exception as e:
            self.logger.warning(f"Could not log model info: {e}")
    
    def unload_model(self, model_name: Optional[str] = None):
        """
        Unload model from memory.
        
        Args:
            model_name: Model to unload (None = unload all)
        """
        if model_name:
            # Unload specific model
            keys_to_remove = [
                k for k in self._models.keys() 
                if k.startswith(model_name)
            ]
            for key in keys_to_remove:
                del self._models[key]
                self.logger.info(f"Unloaded model: {key}")
        else:
            # Unload all models
            self._models.clear()
            self.logger.info("Unloaded all models")
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model_info(self, model_name: Optional[str] = None) -> dict:
        """
        Get information about loaded model(s).
        
        Args:
            model_name: Model name (None = default model)
        
        Returns:
            Dictionary with model information
        """
        model_name = model_name or settings.EMBEDDING_MODEL
        
        try:
            model = self.load_model(model_name)
            
            return {
                "model_name": model_name,
                "embedding_dimension": model.get_sentence_embedding_dimension(),
                "max_seq_length": model.max_seq_length,
                "device": self.device,
                "normalize_embeddings": getattr(model, 'normalize_embeddings', False),
                "loaded": True,
            }
        except Exception as e:
            return {
                "model_name": model_name,
                "error": str(e),
                "loaded": False,
            }
    
    def list_loaded_models(self) -> list:
        """
        List all currently loaded models.
        
        Returns:
            List of loaded model identifiers
        """
        return list(self._models.keys())
    
    def get_device_info(self) -> dict:
        """
        Get information about available devices.
        
        Returns:
            Dictionary with device information
        """
        info = {
            "current_device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_device_count": torch.cuda.device_count(),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**2:.1f} MB",
                "cuda_memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**2:.1f} MB",
            })
        
        return info
    
    def optimize_model(
        self,
        model: SentenceTransformer,
        use_half_precision: bool = False,
        compile_model: bool = False
    ) -> SentenceTransformer:
        """
        Optimize model for faster inference.
        
        Args:
            model: Model to optimize
            use_half_precision: Use FP16 precision (GPU only)
            compile_model: Compile with torch.compile (PyTorch 2.0+)
        
        Returns:
            Optimized model
        """
        try:
            # Half precision (FP16)
            if use_half_precision and self.device == "cuda":
                model = model.half()
                self.logger.info("Converted model to FP16 precision")
            
            # Torch compile (PyTorch 2.0+)
            if compile_model:
                try:
                    import torch._dynamo
                    model = torch.compile(model)
                    self.logger.info("Compiled model with torch.compile")
                except Exception as e:
                    self.logger.warning(f"Could not compile model: {e}")
            
            return model
        
        except Exception as e:
            self.logger.warning(f"Model optimization failed: {e}")
            return model
    
    def warmup(self, model: Optional[SentenceTransformer] = None, num_samples: int = 5):
        """
        Warm up model with dummy inputs.
        
        Args:
            model: Model to warm up (None = default model)
            num_samples: Number of warmup samples
        """
        if model is None:
            model = self.load_model()
        
        self.logger.info("Warming up model...")
        
        # Create dummy inputs
        dummy_texts = [
            f"This is a warmup sentence number {i}" 
            for i in range(num_samples)
        ]
        
        try:
            # Run inference
            _ = model.encode(dummy_texts, show_progress_bar=False)
            self.logger.info("Model warmup complete")
        except Exception as e:
            self.logger.warning(f"Model warmup failed: {e}")


# Global loader instance
_loader = None


def get_model_loader() -> ModelLoader:
    """
    Get global ModelLoader instance (singleton).
    
    Returns:
        ModelLoader instance
    """
    global _loader
    if _loader is None:
        _loader = ModelLoader()
    return _loader


def load_embedding_model(
    model_name: Optional[str] = None,
    device: Optional[str] = None
) -> SentenceTransformer:
    """
    Convenience function to load embedding model.
    
    Args:
        model_name: Model name
        device: Device
    
    Returns:
        Loaded model
    """
    loader = get_model_loader()
    return loader.load_model(model_name, device)


if __name__ == "__main__":
    # Test model loader
    print("=== Model Loader Tests ===\n")
    
    loader = ModelLoader()
    
    # Test 1: Device detection
    print("Test 1: Device detection")
    device_info = loader.get_device_info()
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    print()
    
    # Test 2: Load default model
    print("Test 2: Load default model")
    try:
        model = loader.load_model()
        print(f"  ✓ Model loaded successfully")
        
        info = loader.get_model_info()
        print(f"  Model: {info['model_name']}")
        print(f"  Dimension: {info['embedding_dimension']}")
        print(f"  Max length: {info['max_seq_length']}")
        print(f"  Device: {info['device']}")
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
    print()
    
    # Test 3: List loaded models
    print("Test 3: List loaded models")
    loaded = loader.list_loaded_models()
    print(f"  Loaded models: {loaded}")
    print()
    
    # Test 4: Model warmup
    print("Test 4: Model warmup")
    try:
        loader.warmup(model, num_samples=3)
        print("  ✓ Warmup complete")
    except Exception as e:
        print(f"  ✗ Warmup failed: {e}")
    print()
    
    # Test 5: Test encoding
    print("Test 5: Test encoding")
    try:
        test_texts = ["Hello world", "Test sentence"]
        embeddings = model.encode(test_texts, show_progress_bar=False)
        print(f"  ✓ Encoded {len(test_texts)} texts")
        print(f"  Embedding shape: {embeddings.shape}")
    except Exception as e:
        print(f"  ✗ Encoding failed: {e}")
    print()
    
    # Test 6: Convenience function
    print("Test 6: Convenience function")
    try:
        model2 = load_embedding_model()
        print("  ✓ Loaded via convenience function")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    print()
    
    print("✓ Model loader module created successfully!")