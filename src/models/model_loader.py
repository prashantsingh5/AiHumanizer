"""
Model Loading and Initialization Module
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, 
    AutoModelForSequenceClassification, pipeline,
    T5TokenizerFast, T5ForConditionalGeneration,
    GPT2LMHeadModel, GPT2Tokenizer
)
import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
import spacy
import google.generativeai as genai
from src.config import Config
from src.utils.logger import get_logger

try:
    from language_tool_python import LanguageTool
except ImportError:
    print("⚠️  Installing language-tool-python...")
    import subprocess
    subprocess.check_call(["pip", "install", "language-tool-python"])
    from language_tool_python import LanguageTool

logger = get_logger(__name__)

class ModelLoader:
    """Class to handle loading and managing all required models"""
    
    def __init__(self):
        self.config = Config()
        self._download_nltk_data()
        self._configure_gemini()
        
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
    
    def _configure_gemini(self):
        """Configure Gemini API"""
        genai.configure(api_key=self.config.GEMINI_API_KEY)
    
    def load_all_models(self):
        """Load all required models"""
        logger.info("Loading enhanced models with fallback systems...")
        
        models = {}
        
        # Load spaCy
        models['nlp'] = spacy.load(self.config.SPACY_MODEL)
        
        # Initialize Grammar Checker
        models['grammar_tool'] = self._load_grammar_tool()
        
        # Load paraphrasing models
        paraphrase_models = self._load_paraphrase_models()
        models.update(paraphrase_models)
        
        # Load Gemini
        models['gemini_model'] = self._load_gemini_model()
        
        # Load similarity models
        models['similarity_model'] = SentenceTransformer(self.config.SIMILARITY_MODEL)
        models['advanced_similarity_model'] = self._load_advanced_similarity_model()
        
        logger.info("✅ All models loaded successfully!")
        return models
    
    def _load_grammar_tool(self):
        """Load grammar checking tool"""
        try:
            grammar_tool = LanguageTool(self.config.LANGUAGE_TOOL_LANG)
            logger.info("✅ Grammar checker loaded successfully")
            return grammar_tool
        except Exception as e:
            logger.warning(f"⚠️  Grammar checker failed to load: {e}")
            return None
    
    def _load_paraphrase_models(self):
        """Load all paraphrasing models"""
        models = {}
        
        # Load primary paraphraser
        logger.info("Loading primary paraphraser...")
        models['paraphrase_tokenizer'] = AutoTokenizer.from_pretrained(
            self.config.PARAPHRASE_MODELS['primary']
        )
        models['paraphrase_model'] = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.PARAPHRASE_MODELS['primary']
        )
        
        # Load secondary paraphraser as fallback
        logger.info("Loading secondary paraphraser...")
        try:
            models['secondary_tokenizer'] = AutoTokenizer.from_pretrained(
                self.config.PARAPHRASE_MODELS['secondary']
            )
            models['secondary_model'] = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.PARAPHRASE_MODELS['secondary']
            )
            logger.info("✅ Secondary paraphraser loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️  Secondary paraphraser failed to load: {e}")
            models['secondary_tokenizer'] = None
            models['secondary_model'] = None
        
        # Load BART model
        logger.info("Loading BART paraphraser...")
        try:
            models['bart_tokenizer'] = AutoTokenizer.from_pretrained(
                self.config.PARAPHRASE_MODELS['bart']
            )
            models['bart_model'] = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.PARAPHRASE_MODELS['bart']
            )
            logger.info("✅ BART paraphraser loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️  BART paraphraser failed to load: {e}")
            models['bart_tokenizer'] = None
            models['bart_model'] = None
        
        return models
    
    def _load_gemini_model(self):
        """Load Gemini model with fallback options"""
        try:
            available_models = [model.name for model in genai.list_models()]
            logger.info(f"Available Gemini models: {available_models[:3]}...")
            
            model_names_to_try = [
                'gemini-1.5-flash',
                'gemini-1.5-pro', 
                'gemini-pro',
                'models/gemini-1.5-flash',
                'models/gemini-1.5-pro'
            ]
            
            for model_name in model_names_to_try:
                try:
                    gemini_model = genai.GenerativeModel(model_name)
                    test_response = gemini_model.generate_content("Hello")
                    logger.info(f"✅ Gemini API connected successfully with model: {model_name}")
                    return gemini_model
                except Exception as e:
                    logger.warning(f"⚠️  Failed to connect with {model_name}: {str(e)[:100]}...")
                    continue
            
            logger.warning("⚠️  All Gemini models failed to connect, continuing without Gemini")
            return None
            
        except Exception as e:
            logger.error(f"⚠️  Gemini API setup failed: {e}")
            return None
    
    def _load_advanced_similarity_model(self):
        """Load advanced similarity model with fallback"""
        logger.info("Loading advanced similarity model...")
        try:
            advanced_model = SentenceTransformer(self.config.ADVANCED_SIMILARITY_MODEL)
            logger.info("✅ Advanced similarity model loaded successfully")
            return advanced_model
        except Exception as e:
            logger.warning(f"⚠️  Using basic similarity model: {e}")
            return SentenceTransformer(self.config.SIMILARITY_MODEL)
