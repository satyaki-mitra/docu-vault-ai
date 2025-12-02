# DEPENDENCIES
import math
from typing import Any
from typing import Dict
from typing import Optional
from config.settings import get_settings
from config.logging_config import get_logger
from utils.error_handler import handle_errors
from config.models import TemperatureStrategy
from utils.error_handler import TemperatureControlError


# Setup Settings and Logging
settings = get_settings()
logger   = get_logger(__name__)


class TemperatureController:
    """
    Intelligent temperature control for LLM generation: Implements adaptive temperature strategies based on query type, complexity, and desired output characteristics
    """
    def __init__(self, base_temperature: float = None, strategy: TemperatureStrategy = None):
        """
        Initialize temperature controller
        
        Arguments:
        ----------
            base_temperature { float } : Base temperature value (default from settings)
            
            strategy          { str }  : Temperature control strategy
        """
        self.logger           = logger
        self.settings         = get_settings()
        self.base_temperature = base_temperature or self.settings.DEFAULT_TEMPERATURE
        self.strategy         = strategy or TemperatureStrategy.ADAPTIVE
        
        # Validate base temperature
        if not (0.0 <= self.base_temperature <= 1.0):
            raise TemperatureControlError(f"Temperature must be between 0 and 1: {self.base_temperature}")
        
        # Strategy configurations
        self.strategy_configs = {TemperatureStrategy.FIXED       : {"description" : "Fixed temperature for all queries", "range" : (0.0, 1.0)},
                                 TemperatureStrategy.ADAPTIVE    : {"description" : "Adapt temperature based on query complexity", "range" : (0.1, 0.8), "complexity_threshold" : 0.6},
                                 TemperatureStrategy.CONFIDENCE  : {"description" : "Adjust temperature based on retrieval confidence", "range" : (0.1, 0.9), "high_confidence_temp" : 0.1, "low_confidence_temp" : 0.7},
                                 TemperatureStrategy.PROGRESSIVE : {"description" : "Progressively increase temperature for creative tasks", "range" : (0.1, 0.9), "creative_threshold" : 0.7}
                                }
        
        self.logger.info(f"Initialized TemperatureController: base={self.base_temperature}, strategy={self.strategy}")
    

    def get_temperature(self, query: str = "", context: str = "", retrieval_scores: Optional[list] = None, query_type: str = "qa") -> float:
        """
        Get appropriate temperature for generation
        
        Arguments:
        ----------
            query            { str }  : User query
            
            context          { str }  : Retrieved context
            
            retrieval_scores { list } : Scores of retrieved chunks
            
            query_type       { str }  : Type of query ('qa', 'creative', 'analytical', 'summary')
        
        Returns:
        --------
                   { float }          : Temperature value (0.0 - 1.0)
        """
        if (self.strategy == TemperatureStrategy.FIXED):
            return self._fixed_temperature()
        
        elif (self.strategy == TemperatureStrategy.ADAPTIVE):
            return self._adaptive_temperature(query      = query, 
                                              context    = context, 
                                              query_type = query_type,
                                             )
        
        elif (self.strategy == TemperatureStrategy.CONFIDENCE):
            return self._confidence_based_temperature(retrieval_scores = retrieval_scores, 
                                                      query_type       = query_type,
                                                     )
        
        elif (self.strategy == TemperatureStrategy.PROGRESSIVE):
            return self._progressive_temperature(query_type = query_type, 
                                                 query      = query,
                                                )
        
        else:
            self.logger.warning(f"Unknown strategy: {self.strategy}, using fixed")
            return self.base_temperature
    

    def _fixed_temperature(self) -> float:
        """
        Fixed temperature strategy
        """
        return self.base_temperature
    

    def _adaptive_temperature(self, query: str, context: str, query_type: str) -> float:
        """
        Adaptive temperature based on query complexity and type
        """
        base_temp        = self.base_temperature
        
        # Adjust based on query type
        type_adjustments = {"qa"         : -0.2,      # More deterministic for Q&A
                            "creative"   : 0.3,       # More creative for creative tasks
                            "analytical" : -0.1,      # Slightly deterministic for analysis
                            "summary"    : -0.15,     # Deterministic for summarization
                            "comparison" : 0.1,       # Slightly creative for comparisons
                           }
        
        adjustment       = type_adjustments.get(query_type, 0.0)
        temp             = base_temp + adjustment
        
        # Adjust based on query complexity
        complexity       = self._calculate_query_complexity(query = query)
        
        if (complexity > 0.7): 
            # High complexity
            temp += 0.1

        elif (complexity < 0.3):  
            # Low complexity
            temp -= 0.1
        
        # Adjust based on context quality
        if context:
            context_quality = self._calculate_context_quality(context = context)

            # Poor context
            if (context_quality < 0.5):  
                # More creative when context is poor
                temp += 0.15  
        
        return self._clamp_temperature(temperature = temp)
    

    def _confidence_based_temperature(self, retrieval_scores: Optional[list], query_type: str) -> float:
        """
        Temperature based on retrieval confidence
        """
        if not retrieval_scores:
            self.logger.debug("No retrieval scores, using base temperature")
            return self.base_temperature
        
        # Calculate average confidence
        avg_confidence = sum(retrieval_scores) / len(retrieval_scores)
        
        config         = self.strategy_configs[TemperatureStrategy.CONFIDENCE]
        high_temp      = config["high_confidence_temp"]
        low_temp       = config["low_confidence_temp"]
        
        # High confidence -> low temperature (deterministic) & Low confidence -> high temperature (creative)
        if (avg_confidence > 0.8):
            temperature = high_temp

        elif (avg_confidence < 0.3):
            temperature = low_temp

        else:
            # Linear interpolation between high and low temps
            normalized_confidence = (avg_confidence - 0.3) / (0.8 - 0.3)
            temperature           = high_temp + (low_temp - high_temp) * (1 - normalized_confidence)
        
        # Adjust for query type
        if (query_type == "creative"):
            temperature = min(0.9, temperature + 0.2)

        elif (query_type == "qa"):
            temperature = max(0.1, temperature - 0.1)
        
        return self._clamp_temperature(temperature = temperature)
    

    def _progressive_temperature(self, query_type: str, query: str) -> float:
        """
        Progressive temperature based on task requirements
        """
        base_temp = self.base_temperature
        
        # Task-based progression
        if (query_type == "creative"):
            # High creativity
            return self._clamp_temperature(temperature = 0.8)  
        
        elif (query_type == "analytical"):
            # Balanced
            return self._clamp_temperature(temperature = 0.3)  
        
        elif (query_type == "qa"):
            # For factual Q&A, use lower temperature
            if self._is_factual_query(query):
                return self._clamp_temperature(temperature = 0.1)
            
            else:
                return self._clamp_temperature(temperature = 0.4)
        
        elif (query_type == "summary"):
            # Deterministic summaries
            return self._clamp_temperature(temperature = 0.2)  
        
        else:
            return self._clamp_temperature(temperature = base_temp)
    

    def _calculate_query_complexity(self, query: str) -> float:
        """
        Simple, predictable complexity score
        """
        if not query:
            return 0.5
        
        # Count words and questions
        words       = len(query.split())
        has_why_how = any(word in query.lower() for word in ['why', 'how', 'explain'])
        has_compare = any(word in query.lower() for word in ['compare', 'contrast', 'difference'])
        
        # Simple rules
        if has_compare:
            # Complex
            return 0.8  
        
        elif (has_why_how and( words > 15)):
            return 0.7
        
        elif words > 20:
            return 0.6
        
        else:
            # Simple
            return 0.3  
    

    def _calculate_context_quality(self, context: str) -> float:
        """
        Calculate context quality (0.0 - 1.0)
        """
        if not context:
            return 0.0
        
        factors          = list()
        
        # Length factor (adequate context)
        words            = len(context.split())

        # Normalize
        length_factor    = min(words / 500, 1.0)  

        factors.append(length_factor)
        
        # Diversity factor (multiple sources/citations)
        citation_count   = context.count('[')
        diversity_factor = min(citation_count / 5, 1.0)
        
        factors.append(diversity_factor)
        
        # Coherence factor (simple measure)
        sentence_count   = context.count('.')
        
        if (sentence_count > 0):
            avg_sentence_length = words / sentence_count
            # Ideal ~20 words/sentence
            coherence_factor    = 1.0 - min(abs(avg_sentence_length - 20) / 50, 1.0)  

            factors.append(coherence_factor)
        
        return sum(factors) / len(factors)
    

    def _is_factual_query(self, query: str) -> bool:
        """
        Check if query is factual (requires precise answers)
        """
        factual_indicators = ['what is', 'who is', 'when did', 'where is', 'how many', 'how much', 'definition of', 'meaning of', 'calculate', 'number of']
        
        query_lower        = query.lower()

        return any(indicator in query_lower for indicator in factual_indicators)
    

    def _clamp_temperature(self, temperature: float) -> float:
        """
        Clamp temperature to valid range
        """
        strategy_config = self.strategy_configs.get(self.strategy, {})
        temp_range      = strategy_config.get("range", (0.0, 1.0))
        
        clamped         = max(temp_range[0], min(temperature, temp_range[1]))

        # Round to 2 decimal places
        clamped         = round(clamped, 2)  
        
        return clamped
    

    def get_temperature_parameters(self, temperature: float) -> Dict[str, Any]:
        """
        Get additional parameters based on temperature
        
        Arguments:
        ----------
            temperature { float } : Temperature value
        
        Returns:
        --------
                   { dict }       : Additional generation parameters
        """
        params = {"temperature" : temperature,
                  "top_p"       : 0.9,
                 }
        
        # Adjust top_p based on temperature
        if (temperature < 0.3):
            # Broader distribution for low temp
            params["top_p"] = 0.95  

        elif (temperature > 0.7):
            # Narrower distribution for high temp
            params["top_p"] = 0.7  
        
        # Adjust presence_penalty based on temperature
        if (temperature > 0.5):
            # Encourage novelty for creative tasks
            params["presence_penalty"] = 0.1  

        else:
            params["presence_penalty"] = 0.0
        
        return params
    

    def explain_temperature_choice(self, query: str, context: str, retrieval_scores: list, query_type: str, final_temperature: float) -> Dict[str, Any]:
        """
        Explain why a particular temperature was chosen
        
        Arguments:
        ----------
            query             { str }   : User query
            
            context           { str }   : Retrieved context
            
            retrieval_scores  { list }  : Retrieval scores
            
            query_type        { str }   : Query type
            
            final_temperature { float } : Chosen temperature
        
        Returns:
        --------
                      { dict }          : Explanation dictionary
        """
        explanation = {"strategy"          : self.strategy.value,
                       "final_temperature" : final_temperature,
                       "base_temperature"  : self.base_temperature,
                       "factors"           : {},
                      }
        
        if (self.strategy == TemperatureStrategy.ADAPTIVE):
            complexity             = self._calculate_query_complexity(query = query)
            context_quality        = self._calculate_context_quality(context = context)
            
            explanation["factors"] = {"query_complexity" : round(complexity, 3),
                                      "context_quality"  : round(context_quality, 3),
                                      "query_type"       : query_type,
                                     }
        
        elif (self.strategy == TemperatureStrategy.CONFIDENCE):
            if retrieval_scores:
                avg_confidence         = sum(retrieval_scores) / len(retrieval_scores)
                explanation["factors"] = {"average_retrieval_confidence" : round(avg_confidence, 3),
                                          "query_type"                   : query_type,
                                         }
        
        elif (self.strategy == TemperatureStrategy.PROGRESSIVE):
            is_factual             = self._is_factual_query(query)
            explanation["factors"] = {"query_type"       : query_type,
                                      "is_factual_query" : is_factual,
                                     }
        
        return explanation


# Global temperature controller instance
_temperature_controller = None


def get_temperature_controller() -> TemperatureController:
    """
    Get global temperature controller instance (singleton)
    
    Returns:
    --------
        { TemperatureController }    : TemperatureController instance
    """
    global _temperature_controller
    
    if _temperature_controller is None:
        _temperature_controller = TemperatureController()
    
    return _temperature_controller


@handle_errors(error_type=TemperatureControlError, log_error=True, reraise=False)
def get_adaptive_temperature(query: str = "", context: str = "", retrieval_scores: list = None, query_type: str = "qa") -> float:
    """
    Convenience function for getting adaptive temperature
    
    Arguments:
    ----------
        query            { str }   : User query
        
        context          { str }   : Retrieved context
         
        retrieval_scores { list }  : Retrieval scores
        
        query_type       { str }   : Query type
    
    Returns:
    --------
               { float }           : Temperature value
    """
    controller = get_temperature_controller()
    
    return controller.get_temperature(query            = query, 
                                      context          = context, 
                                      retrieval_scores = retrieval_scores, 
                                      query_type       = query_type,
                                     )