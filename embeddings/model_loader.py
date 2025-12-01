# DEPENDENCIES
import os
import gc
import torch
from typing import Optional
from config.settings import get_settings
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from utils.error_handler import EmbeddingError
from sentence_transformers import SentenceTransformer


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class EmbeddingModelLoader:
    """
    Manages loading and caching of embedding models: Supports multiple models with efficient resource management
    """
    def __init__(self):
        self.logger        = logger
        self._loaded_model = None
        self._model_name   = None
        self._device       = None
        
        # Model cache for multiple models
        self._model_cache  = dict()
    

    @handle_errors(error_type = EmbeddingError, log_error = True, reraise = True)
    def load_model(self, model_name: Optional[str] = None, device: Optional[str] = None, force_reload: bool = False) -> SentenceTransformer:
        """
        Load embedding model with caching and device optimization
        
        Arguments:
        ----------
            model_name   { str }    : Name of model to load (default from settings)
            
            device       { str }    : Device to load on ('cpu', 'cuda', 'mps', 'auto')
            
            force_reload { bool }   : Force reload even if model is cached
        
        Returns:
        --------
            { SentenceTransformer } : Loaded model instance
        
        Raises:
        -------
            EmbeddingError          : If model loading fails
        """
        model_name = model_name or settings.EMBEDDING_MODEL
        device     = self._resolve_device(device)
        
        # Check cache first
        cache_key  = f"{model_name}_{device}"
        
        if ((not force_reload) and (cache_key in self._model_cache)):
            self.logger.debug(f"Using cached model: {cache_key}")
            
            self._loaded_model = self._model_cache[cache_key]
            self._model_name   = model_name
            self._device       = device
            
            return self._loaded_model
        
        try:
            self.logger.info(f"Loading embedding model: {model_name} on device: {device}")
            
            # Load model with optimized settings
            model                        = SentenceTransformer(model_name,
                                                               device       = device,
                                                               cache_folder = os.path.expanduser("~/.cache/sentence_transformers"),
                                                              )
            
            # Model-specific optimizations
            model                        = self._optimize_model(model  = model, 
                                                                device = device,
                                                               )
            
            # Cache the model
            self._model_cache[cache_key] = model
            self._loaded_model           = model
            self._model_name             = model_name
            self._device                 = device
            
            # Log model info
            self._log_model_info(model  = model, 
                                 device = device,
                                )
            
            self.logger.info(f"Successfully loaded model: {model_name}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {repr(e)}")
            raise EmbeddingError(f"Model loading failed: {repr(e)}")
    

    def _resolve_device(self, device: Optional[str] = None) -> str:
        """
        Resolve the best available device
        
        Arguments:
        ----------
            device { str } : Requested device
        
        Returns:
        --------
               { str }     : Actual device to use
        """
        if (device and (device != "auto")):
            return device
        
        # Auto device selection
        if (settings.EMBEDDING_DEVICE != "auto"):
            return settings.EMBEDDING_DEVICE
        
        # Automatic detection
        if torch.cuda.is_available():
            return "cuda"
        
        elif torch.backends.mps.is_available():
            return "mps"
        
        else:
            return "cpu"
    

    def _optimize_model(self, model: SentenceTransformer, device: str) -> SentenceTransformer:
        """
        Apply optimizations to the model
        
        Arguments:
        ----------
            model  { SentenceTransformer } : Model to optimize

            device { str }                 : Device model is on
        
        Returns:
        --------
            { SentenceTransformer }        : Optimized model
        """
        # Enable eval mode for inference
        model.eval()
        
        # GPU optimizations
        if (device == "cuda"):
            # Use half precision for GPU if supported
            try:
                model = model.half()
                self.logger.debug("Enabled half precision for GPU")
            
            except Exception as e:
                self.logger.warning(f"Could not enable half precision: {repr(e)}")
        
        # Disable gradient computation
        for param in model.parameters():
            param.requires_grad = False
        
        return model
    

    def _log_model_info(self, model: SentenceTransformer, device: str):
        """
        Log detailed model information
        
        Arguments:
        ----------
            model  { SentenceTransformer } : Model to log info for

            device { str }                 : Device model is on
        """
        try:
            # Get model architecture info
            if hasattr(model, '_modules'):
                modules = list(model._modules.keys())
            
            else:
                modules = ["unknown"]
            
            # Get embedding dimension
            if hasattr(model, 'get_sentence_embedding_dimension'):
                dimension = model.get_sentence_embedding_dimension()
            
            else:
                dimension = "unknown"
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            
            self.logger.info(f"Model Info: {len(modules)} modules, dimension={dimension}, parameters={total_params:,}, device={device}")
                           
        except Exception as e:
            self.logger.debug(f"Could not get detailed model info: {repr(e)}")
    

    def get_loaded_model(self) -> Optional[SentenceTransformer]:
        """
        Get currently loaded model
        
        Returns:
        --------
            { SentenceTransformer } : Currently loaded model or None
        """
        return self._loaded_model
    

    def get_model_info(self) -> dict:
        """
        Get information about loaded model
        
        Returns:
        --------
            { dict }    : Model information dictionary
        """
        if self._loaded_model is None:
            return {"loaded": False}
        
        info = {"loaded"       : True,
                "model_name"   : self._model_name,
                "device"       : self._device,
                "cache_size"   : len(self._model_cache),
               }
        
        try:
            if hasattr(self._loaded_model, 'get_sentence_embedding_dimension'):
                info["embedding_dimension"] = self._loaded_model.get_sentence_embedding_dimension()
            
            info["model_class"] = type(self._loaded_model).__name__
            
        except Exception as e:
            self.logger.warning(f"Could not get detailed model info: {e}")
        
        return info
    

    def clear_cache(self, model_name: Optional[str] = None):
        """
        Clear model cache
        
        Arguments:
        ----------
            model_name { str } : Specific model to clear (None = all)
        """
        if model_name:
            # Clear specific model from all devices
            keys_to_remove = [k for k in self._model_cache.keys() if k.startswith(model_name)]
            
            for key in keys_to_remove:
                del self._model_cache[key]
            
            self.logger.info(f"Cleared cache for model: {model_name}")
        
        else:
            # Clear all cache
            cache_size = len(self._model_cache)
            self._model_cache.clear()
            
            self.logger.info(f"Cleared all model cache ({cache_size} models)")
    

    def unload_model(self):
        """
        Unload current model and free memory
        """
        if self._loaded_model:
            model_name = self._model_name
            
            # Clear from cache
            if self._model_name and self._device:
                cache_key = f"{self._model_name}_{self._device}"
                self._model_cache.pop(cache_key, None)
            
            # Clear references
            self._loaded_model = None
            self._model_name   = None
            self._device       = None
            
            # Force garbage collection            
            gc.collect()
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.logger.info(f"Unloaded model: {model_name}")


# Global model loader instance
_model_loader = None


def get_model_loader() -> EmbeddingModelLoader:
    """
    Get global model loader instance (singleton)
    
    Returns:
    --------
        { EmbeddingModelLoader } : Model loader instance
    """
    global _model_loader

    if _model_loader is None:
        _model_loader = EmbeddingModelLoader()
    
    return _model_loader


def load_embedding_model(model_name: Optional[str] = None, device: Optional[str] = None) -> SentenceTransformer:
    """
    Convenience function to load embedding model
    
    Arguments:
    ----------
        model_name { str } : Model name
        
        device     { str } : Device
    
    Returns:
    --------
        { SentenceTransformer } : Loaded model
    """
    loader = get_model_loader()

    return loader.load_model(model_name, device)