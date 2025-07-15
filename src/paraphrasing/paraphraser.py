"""
Fixed Advanced Multi-Model Paraphrasing Engine
"""

import torch
import numpy as np
import re
import random
from typing import List
import nltk
from nltk.corpus import wordnet
import textstat
import google.generativeai as genai
from src.utils.logger import get_logger

logger = get_logger(__name__)

class RobustParaphraser:
    """Advanced multi-model paraphrasing engine"""
    
    def __init__(self, primary_tokenizer, primary_model, secondary_tokenizer=None, 
                 secondary_model=None, bart_tokenizer=None, bart_model=None, 
                 gemini_model=None, similarity_model=None, nlp=None):
        self.primary_tokenizer = primary_tokenizer
        self.primary_model = primary_model
        self.secondary_tokenizer = secondary_tokenizer
        self.secondary_model = secondary_model
        self.bart_tokenizer = bart_tokenizer
        self.bart_model = bart_model
        self.gemini_model = gemini_model
        self.similarity_model = similarity_model
        self.nlp = nlp
        
    def generate_paraphrase_candidates(self, text: str, context: str = "", style: str = "casual", 
                                     num_candidates: int = 8) -> List[str]:
        """Generate multiple paraphrase candidates using different strategies with context awareness"""
        candidates = []
        
        # Strategy 1: Primary model with different parameters
        candidates.extend(self._paraphrase_with_model(
            text, self.primary_tokenizer, self.primary_model, "primary", style
        ))
        
        # Strategy 2: Secondary model if available
        if self.secondary_tokenizer and self.secondary_model:
            candidates.extend(self._paraphrase_with_model(
                text, self.secondary_tokenizer, self.secondary_model, "secondary", style
            ))
        
        # Strategy 3: BART model if available
        if self.bart_tokenizer and self.bart_model:
            candidates.extend(self._paraphrase_with_bart(text, style))
        
        # Strategy 4: Gemini API paraphrasing (Fixed)
        if self.gemini_model:
            candidates.extend(self._paraphrase_with_gemini_fixed(text, context, style))
        
        # Strategy 5: Structural variations
        candidates.extend(self._structural_variations(text))
        
        # Strategy 6: Synonym-based variations
        candidates.extend(self._synonym_variations(text))
        
        # Remove duplicates and empty candidates
        candidates = list(set([c.strip() for c in candidates if c.strip() and c.strip() != text]))
        
        return candidates[:num_candidates]
    
    def _paraphrase_with_gemini_fixed(self, text: str, context: str = "", style: str = "casual") -> List[str]:
        """Fixed Gemini API paraphrasing with better error handling"""
        candidates = []
        
        if not self.gemini_model:
            return candidates
        
        # Simplified, more reliable prompts
        prompts = [
            f"Rewrite this sentence in a {style} style: {text}",
            f"Paraphrase this naturally: {text}",
            f"Make this more conversational: {text}"
        ]
        
        for prompt in prompts:
            try:
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=150,
                        candidate_count=1
                    )
                )
                
                if response.text:
                    # Clean up response
                    candidate = response.text.strip()
                    # Remove quotes and unwanted formatting
                    candidate = re.sub(r'^["\'`](.*)["\'`]$', r'\1', candidate)
                    candidate = re.sub(r'^(Here\'s|Here is).*?:', '', candidate, flags=re.IGNORECASE)
                    candidate = candidate.strip()
                    
                    if candidate and candidate != text and len(candidate.split()) > 3:
                        candidates.append(candidate)
                        
            except Exception as e:
                logger.warning(f"⚠️  Gemini paraphrasing attempt failed: {str(e)[:50]}...")
                continue
        
        return candidates
    
    # ...existing code for other methods...
    def _paraphrase_with_bart(self, text: str, style: str = "casual") -> List[str]:
        """Enhanced BART paraphrasing with better prompts"""
        candidates = []
        
        if not self.bart_tokenizer or not self.bart_model:
            return candidates
        
        try:
            prompts = [
                f"Rewrite: {text}",
                f"Paraphrase: {text}",
                text
            ]
            
            for prompt in prompts:
                inputs = self.bart_tokenizer(
                    prompt, return_tensors="pt", max_length=512, truncation=True
                )
                
                with torch.no_grad():
                    outputs = self.bart_model.generate(
                        **inputs,
                        max_length=min(150, len(text.split()) * 2),
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1,
                        pad_token_id=self.bart_tokenizer.eos_token_id
                    )
                
                candidate = self.bart_tokenizer.decode(outputs[0], skip_special_tokens=True)
                candidate = re.sub(r'^(rewrite|paraphrase):\s*', '', candidate, flags=re.IGNORECASE)
                candidate = candidate.strip()
                
                if candidate and candidate != text and len(candidate.split()) > 3:
                    candidates.append(candidate)
                        
        except Exception as e:
            logger.error(f"⚠️  BART paraphrasing failed: {e}")
        
        return candidates
    
    def _paraphrase_with_model(self, text: str, tokenizer, model, model_type: str, style: str = "casual") -> List[str]:
        """Enhanced model paraphrasing with better error handling"""
        candidates = []
        
        prompts = [
            f"paraphrase: {text}",
            f"rewrite: {text}",
            f"rephrase: {text}"
        ]
        
        for prompt in prompts:
            try:
                inputs = tokenizer(
                    prompt, return_tensors="pt", max_length=512, truncation=True
                )
                
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_length=min(200, len(text.split()) * 3),
                        num_return_sequences=1,
                        temperature=0.8,
                        do_sample=True,
                        top_p=0.9,
                        top_k=50,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
                
                candidate = tokenizer.decode(outputs[0], skip_special_tokens=True)
                candidate = re.sub(r'^(paraphrase|rewrite|rephrase):\s*', '', candidate, flags=re.IGNORECASE)
                candidate = candidate.strip()
                
                if candidate and candidate != text and len(candidate.split()) > 3:
                    candidates.append(candidate)
                        
            except Exception as e:
                logger.error(f"⚠️  {model_type} model paraphrasing failed: {e}")
                continue
        
        return candidates
    
    def _structural_variations(self, text: str) -> List[str]:
        """Enhanced structural variations"""
        candidates = []
        if not self.nlp:
            return candidates
            
        doc = self.nlp(text)
        
        # Split compound sentences
        if len(list(doc.sents)) == 1 and len([token for token in doc if token.text in [',', 'and', 'but', 'or']]) > 1:
            sentence = text.strip()
            
            conjunctions = [', and ', ', but ', ', or ', '. And ', '. But ', '. Or ']
            for conj in conjunctions:
                if conj.lower() in sentence.lower():
                    parts = sentence.split(conj)
                    if len(parts) == 2:
                        candidates.append(f"{parts[0].strip()}. {parts[1].strip()}")
                        candidates.append(f"{parts[1].strip()}, while {parts[0].strip().lower()}")
        
        # Rearrange clauses
        sentences = [sent.text.strip() for sent in doc.sents]
        if len(sentences) > 1:
            candidates.append(' '.join(reversed(sentences)))
        
        return candidates
    
    def _synonym_variations(self, text: str) -> List[str]:
        """Enhanced synonym variations with context awareness"""
        candidates = []
        if not self.nlp:
            return candidates
            
        doc = self.nlp(text)
        
        key_words = [token for token in doc if token.pos_ in ['NOUN', 'VERB', 'ADJ', 'ADV'] and not token.is_stop]
        
        if len(key_words) > 0:
            for _ in range(3):
                text_copy = text
                words_to_replace = random.sample(key_words, min(2, len(key_words)))
                
                for word in words_to_replace:
                    synonyms = self._get_synonyms(word.text, word.pos_)
                    if synonyms:
                        synonym = random.choice(synonyms)
                        text_copy = re.sub(r'\b' + re.escape(word.text) + r'\b', synonym, text_copy, flags=re.IGNORECASE)
                
                if text_copy != text:
                    candidates.append(text_copy)
        
        return candidates
    
    def _get_synonyms(self, word: str, pos: str = None) -> List[str]:
        """Enhanced synonym extraction with POS awareness"""
        synonyms = set()
        
        try:
            pos_map = {
                'NOUN': 'n', 'VERB': 'v', 'ADJ': 'a', 'ADV': 'r'
            }
            wn_pos = pos_map.get(pos, 'n')
            
            for syn in wordnet.synsets(word, pos=wn_pos):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower() and len(synonym.split()) <= 2:
                        synonyms.add(synonym)
        except:
            pass
        
        return list(synonyms)[:3]
    
    def select_best_candidate(self, original: str, candidates: List[str], context: str = "") -> str:
        """More conservative candidate selection"""
        if not candidates or not self.similarity_model:
            return original
        
        best_candidate = original
        best_score = -1
        
        for candidate in candidates:
            try:
                # Check for basic quality issues first
                if len(candidate.split()) < 3 or not candidate.strip():
                    continue
                    
                # Check for incomplete sentences
                words = candidate.split()
                last_word = re.sub(r'[^\w]', '', words[-1]).lower()
                if last_word in ['the', 'a', 'an', 'of', 'to', 'in', 'for', 'with', 'by']:
                    continue
                
                # Calculate semantic similarity
                similarity = self.similarity_model.encode([original, candidate])
                semantic_sim = np.dot(similarity[0], similarity[1]) / (
                    np.linalg.norm(similarity[0]) * np.linalg.norm(similarity[1])
                )
                
                # Higher similarity threshold for better quality
                if semantic_sim < 0.85:
                    continue
                
                # Calculate diversity but with less weight
                word_overlap = len(set(original.lower().split()) & set(candidate.lower().split()))
                total_words = len(set(original.lower().split()) | set(candidate.lower().split()))
                diversity = 1 - (word_overlap / total_words) if total_words > 0 else 0
                
                # More conservative scoring - prioritize similarity
                score = semantic_sim * 0.8 + diversity * 0.2
                
                if score > best_score and semantic_sim > 0.85:
                    best_score = score
                    best_candidate = candidate
                    
            except Exception as e:
                logger.error(f"⚠️  Error evaluating candidate: {e}")
                continue
        
        return best_candidate
