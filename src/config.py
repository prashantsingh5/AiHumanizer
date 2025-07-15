"""
Configuration settings for AI Humanizer
"""

import os

class Config:
    """Configuration class for AI Humanizer"""
    
    # API Keys
    GEMINI_API_KEY = "Enter_your_key_here"
    
    # Model configurations
    PARAPHRASE_MODELS = {
        'primary': "Vamsi/T5_Paraphrase_Paws",
        'secondary': "ramsrigouthamg/t5_paraphraser",
        'tertiary': "tuner007/pegasus_paraphrase",
        'bart': "facebook/bart-large-cnn",
        'gemini': "gemini-1.5-flash"
    }
    
    # Similarity models
    SIMILARITY_MODEL = 'all-MiniLM-L6-v2'
    ADVANCED_SIMILARITY_MODEL = 'all-mpnet-base-v2'
    
    # SpaCy model
    SPACY_MODEL = "en_core_web_sm"
    
    # Language tool
    LANGUAGE_TOOL_LANG = 'en-US'
    
    # Processing parameters
    DEFAULT_STYLE = "casual"
    DEFAULT_PERSONA = "casual"
    DEFAULT_PRESERVE_THRESHOLD = 0.8
    MAX_CANDIDATES = 8
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
