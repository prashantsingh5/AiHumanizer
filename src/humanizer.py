"""
Enhanced Humanization Engine with Advanced Features
"""

import numpy as np
import re
import random
import textstat
from typing import Dict, List, Optional
from src.models.model_loader import ModelLoader
from src.patterns.pattern_library import EnhancedPatternLibrary
from src.paraphrasing.paraphraser import RobustParaphraser
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SuperiorHumanizer:
    """Main humanizer class with advanced features"""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.models = self.model_loader.load_all_models()
        self.patterns = EnhancedPatternLibrary()
        
        self.paraphraser = RobustParaphraser(
            self.models.get('paraphrase_tokenizer'),
            self.models.get('paraphrase_model'),
            self.models.get('secondary_tokenizer'),
            self.models.get('secondary_model'),
            self.models.get('bart_tokenizer'),
            self.models.get('bart_model'),
            self.models.get('gemini_model'),
            self.models.get('advanced_similarity_model'),
            self.models.get('nlp')
        )
        
        self.similarity_model = self.models.get('advanced_similarity_model')
        self.nlp = self.models.get('nlp')
        self.grammar_tool = self.models.get('grammar_tool')
        
        # Enhanced AI detection patterns
        self.ai_patterns = [
            r'\b(furthermore|moreover|additionally|consequently|therefore|thus|however|nevertheless|nonetheless)\b',
            r'\b(comprehensive|utilize|facilitate|implement|demonstrate|indicate|establish)\b',
            r'\b(significant|substantial|numerous|various|multitude|endeavor|attempt|obtain)\b',
            r'\b(in conclusion|to summarize|in summary|to conclude|in summary|overall)\b',
            r'\b(it is important to note|it should be noted|it is worth noting|notably)\b',
            r'\b(in order to|with regard to|in terms of|for the purpose of|with respect to)\b',
            r'\b(predominantly|approximately|methodology|optimal|paramount)\b',
            r'\b(pertaining to|in regard to|subsequent to|prior to|in accordance with)\b',
            r'\b(manifests|exhibits|encompasses|constitutes|exemplifies)\b',
            r'\b(delve into|dive deep|explore thoroughly|examine closely)\b',
            r'\b(cutting-edge|state-of-the-art|revolutionary|groundbreaking)\b'
        ]
    
    def detect_topic(self, text: str) -> str:
        """Simple topic detection based on keywords"""
        text_lower = text.lower()
        
        tech_keywords = ['technology', 'digital', 'software', 'computer', 'ai', 'algorithm', 'data']
        health_keywords = ['health', 'medical', 'doctor', 'patient', 'treatment', 'medicine', 'wellness']
        edu_keywords = ['education', 'learning', 'student', 'teacher', 'school', 'knowledge', 'study']
        
        if any(keyword in text_lower for keyword in tech_keywords):
            return 'technology'
        elif any(keyword in text_lower for keyword in health_keywords):
            return 'health'
        elif any(keyword in text_lower for keyword in edu_keywords):
            return 'education'
        else:
            return 'general'
    
    def check_grammar(self, text: str) -> str:
        """Check and fix grammar issues"""
        if not self.grammar_tool:
            return text
        
        try:
            matches = self.grammar_tool.check(text)
            corrected = self.grammar_tool.correct(text)
            return corrected
        except Exception as e:
            logger.error(f"⚠️  Grammar check failed: {e}")
            return text
    
    def calculate_ai_score(self, text: str) -> float:
        """Enhanced AI detection scoring with more sophisticated metrics"""
        score = 0.0
        text_lower = text.lower()
        
        # Pattern matching with weights
        pattern_weights = [0.15, 0.12, 0.10, 0.20, 0.18, 0.15, 0.12, 0.15, 0.10, 0.08, 0.05]
        for i, pattern in enumerate(self.ai_patterns):
            matches = len(re.findall(pattern, text_lower))
            weight = pattern_weights[i] if i < len(pattern_weights) else 0.10
            score += matches * weight
        
        # ...existing code for sentence structure analysis...
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        if len(sentences) > 1:
            lengths = [len(s.split()) for s in sentences]
            avg_length = np.mean(lengths)
            std_length = np.std(lengths)
            
            if std_length < 4 and avg_length > 15:
                score += 0.20
            
            if avg_length > 30:
                score += 0.15
            
            if self.nlp:
                structures = []
                for sent in sentences:
                    doc = self.nlp(sent)
                    pos_pattern = [token.pos_ for token in doc]
                    structures.append(tuple(pos_pattern[:5]))
                
                if len(set(structures)) < len(structures) * 0.7:
                    score += 0.10
        
        # ...existing code for other scoring factors...
        total_words = len(text.split())
        contractions = len(re.findall(r"\b\w+'[a-z]+\b", text))
        if total_words > 30 and contractions < 2:
            score += 0.12
        
        words = [w.lower() for w in re.findall(r'\b\w+\b', text)]
        if len(words) > 20:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio > 0.85:
                score += 0.10
            elif unique_ratio < 0.50:
                score += 0.08
        
        try:
            flesch_score = textstat.flesch_reading_ease(text)
            if flesch_score < 20 or flesch_score > 95:
                score += 0.12
        except:
            pass
        
        ai_phrases = [
            r'it is important to understand that',
            r'it should be noted that',
            r'one might argue that',
            r'it is worth mentioning',
            r'in today\'s digital age',
            r'with the advent of',
            r'as we move forward'
        ]
        
        for phrase_pattern in ai_phrases:
            if re.search(phrase_pattern, text_lower):
                score += 0.08
        
        return min(score, 1.0)
    
    def advanced_humanization(self, text: str, style: str = "casual", persona: str = "casual", 
                            preserve_threshold: float = 0.8, topic: str = None) -> str:
        """Main humanization function with style and persona control"""
        
        original_text = text.strip()
        detected_topic = topic or self.detect_topic(original_text)
        
        print(f"Original text: {len(original_text.split())} words")
        print(f"Detected topic: {detected_topic}")
        print(f"Style: {style}, Persona: {persona}")
        print(f"Original AI score: {self.calculate_ai_score(original_text):.3f}")
        
        # Step 1: Paragraph-level restructuring with context
        print("\n1. Context-aware paragraph restructuring...")
        restructured = self.paragraph_level_restructuring(original_text, style, detected_topic)
        
        # Step 2: Check content preservation
        similarity = self.calculate_similarity(original_text, restructured)
        print(f"Content similarity after restructuring: {similarity:.3f}")
        
        if similarity < preserve_threshold:
            print("⚠️  Similarity too low, using sentence-level processing only")
            restructured = self.process_paragraph(original_text, style, persona, detected_topic)
        
        # Step 3: Grammar checking
        print("\n2. Grammar checking and correction...")
        grammar_checked = self.check_grammar(restructured)
        
        # Step 4: Final post-processing
        print("\n3. Final post-processing...")
        final_text = self.post_process_text(grammar_checked)
        
        # Final metrics
        final_ai_score = self.calculate_ai_score(final_text)
        final_similarity = self.calculate_similarity(original_text, final_text)
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS:")
        print(f"Original length: {len(original_text.split())} words")
        print(f"Final length: {len(final_text.split())} words")
        print(f"Content similarity: {final_similarity:.3f}")
        print(f"Original AI score: {self.calculate_ai_score(original_text):.3f}")
        print(f"Final AI score: {final_ai_score:.3f}")
        print(f"Improvement: {self.calculate_ai_score(original_text) - final_ai_score:.3f}")
        print(f"Topic: {detected_topic}")
        print(f"Style: {style}")
        print(f"{'='*60}")
        
        return final_text
    
    # ...existing code for other methods...
    def paragraph_level_restructuring(self, text: str, style: str = "casual", topic: str = "general") -> str:
        """Enhanced paragraph restructuring with style and topic awareness"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        if len(paragraphs) < 2:
            sentences = [s.strip() for s in text.split('.') if s.strip()]
            if len(sentences) > 4 and self.nlp:
                doc = self.nlp(text)
                entities = [ent.text for ent in doc.ents]
                
                split_points = []
                for i, sent in enumerate(sentences[1:], 1):
                    sent_doc = self.nlp(sent)
                    sent_entities = [ent.text for ent in sent_doc.ents]
                    
                    if len(set(entities) & set(sent_entities)) < len(entities) * 0.3:
                        split_points.append(i)
                
                if split_points:
                    split_point = split_points[0]
                    paragraphs = [
                        '. '.join(sentences[:split_point]) + '.',
                        '. '.join(sentences[split_point:]) + '.'
                    ]
                else:
                    mid_point = len(sentences) // 2
                    paragraphs = [
                        '. '.join(sentences[:mid_point]) + '.',
                        '. '.join(sentences[mid_point:]) + '.'
                    ]
        
        processed_paragraphs = []
        for i, paragraph in enumerate(paragraphs):
            context = ' '.join(processed_paragraphs) if processed_paragraphs else ""
            processed = self.process_paragraph(paragraph, style, "casual", topic, context)
            processed_paragraphs.append(processed)
        
        return '\n\n'.join(processed_paragraphs)
    
    def process_paragraph(self, paragraph: str, style: str = "casual", persona: str = "casual", 
                         topic: str = "general", context: str = "") -> str:
        """Enhanced paragraph processing with full context awareness"""
        sentences = [s.strip() for s in paragraph.split('.') if s.strip()]
        processed_sentences = []
        
        for i, sentence in enumerate(sentences):
            if len(sentence.split()) < 4:
                processed_sentences.append(sentence)
                continue
            
            candidates = self.paraphraser.generate_paraphrase_candidates(
                sentence, context, style
            )
            
            best_sentence = self.paraphraser.select_best_candidate(sentence, candidates, context)
            
            best_sentence = self.apply_comprehensive_patterns(
                best_sentence, i, len(sentences), style, persona, topic
            )
            
            processed_sentences.append(best_sentence)
            context += " " + best_sentence
        
        return '. '.join(processed_sentences) + '.'
    
    def apply_comprehensive_patterns(self, sentence: str, position: int, total_sentences: int, 
                                   style: str = "casual", persona: str = "casual", topic: str = "general") -> str:
        """Enhanced pattern application with reduced over-processing"""
        
        # 1. Apply contractions (keep this)
        for formal, casual in self.patterns.contractions.items():
            sentence = re.sub(r'\b' + re.escape(formal) + r'\b', casual, sentence, flags=re.IGNORECASE)
        
        # 2. Replace formal words with style-appropriate alternatives (keep but limit)
        words = sentence.split()
        replaced_count = 0
        for i, word in enumerate(words):
            if replaced_count >= 2:  # Limit replacements per sentence
                break
            clean_word = re.sub(r'[^\w]', '', word.lower())
            replacement = self.patterns.get_style_appropriate_replacement(clean_word, style)
            
            if replacement != clean_word:
                if word[0].isupper():
                    replacement = replacement.capitalize()
                words[i] = re.sub(re.escape(clean_word), replacement, word, flags=re.IGNORECASE)
                replaced_count += 1
        
        sentence = ' '.join(words)
        
        # 3. Add persona starters - REDUCED frequency
        if position == 0 and random.random() < 0.15:  # Only first sentence, 15% chance
            starter = self.patterns.get_persona_starter(persona)
            sentence = f"{starter} {sentence.lower()}"
        
        # 4. Add natural fillers - MUCH REDUCED frequency
        filler_probability = {'casual': 0.05, 'conversational': 0.08, 'professional': 0.02}.get(style, 0.03)
        if random.random() < filler_probability and len(sentence.split()) > 10:
            words = sentence.split()
            filler = random.choice(['actually', 'really', 'basically'])  # Use milder fillers
            insert_pos = random.randint(3, len(words) - 3)
            words.insert(insert_pos, filler)
            sentence = ' '.join(words)
        
        # 5. Topic-specific vocabulary (keep but limit to one per sentence)
        if topic in self.patterns.topic_vocabularies and random.random() < 0.3:
            topic_vocab = self.patterns.topic_vocabularies[topic]
            for formal_word, replacements in list(topic_vocab.items())[:1]:  # Only try first match
                if formal_word in sentence.lower():
                    replacement = random.choice(replacements)
                    sentence = re.sub(r'\b' + re.escape(formal_word) + r'\b', replacement, 
                                    sentence, flags=re.IGNORECASE)
                    break
        
        return sentence
    
    def post_process_text(self, text: str) -> str:
        """Enhanced post-processing with better quality control"""
        
        # Fix spacing issues
        text = re.sub(r'\s+', ' ', text)
        
        # Remove excessive filler insertions
        text = re.sub(r'\s*-\s*[^-]*\s*-\s*', ' ', text)  # Remove all "- filler -" patterns
        
        # Fix punctuation spacing
        text = re.sub(r'\s*,\s*', ', ', text)
        text = re.sub(r'\s*\.\s*', '. ', text)
        
        # Fix double periods
        text = re.sub(r'\.+', '.', text)
        
        # Remove incomplete sentences (sentences ending with "the." or similar)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        clean_sentences = []
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
                
            # Skip sentences that end with articles or prepositions only
            words = sent.split()
            if len(words) > 0:
                last_word = re.sub(r'[^\w]', '', words[-1]).lower()
                if last_word in ['the', 'a', 'an', 'of', 'to', 'in', 'for', 'with', 'by']:
                    continue  # Skip incomplete sentences
                    
            # Skip very short or malformed sentences
            if len(words) < 3:
                continue
                
            clean_sentences.append(sent)
        
        text = ' '.join(clean_sentences)
        
        # Ensure proper capitalization
        sentences = []
        for sent in re.split(r'(?<=[.!?])\s+', text):
            if sent and len(sent) > 1:
                sent = sent[0].upper() + sent[1:]
                sentences.append(sent)
        
        text = ' '.join(sentences)
        
        # Fix "i" to "I"
        text = re.sub(r'\bi\b', 'I', text)
        
        # Remove repetitive sentences - FIXED
        sentences = text.split('. ')
        unique_sentences = []
        seen_content = []  # Changed to list instead of set
        
        for sent in sentences:
            # Normalize for comparison
            normalized = re.sub(r'[^\w\s]', '', sent.lower()).strip()
            key_words = set(normalized.split())
            
            # Check if this sentence is too similar to previous ones
            is_similar = False
            for seen_words in seen_content:
                if len(key_words & seen_words) > len(key_words) * 0.7:  # 70% word overlap
                    is_similar = True
                    break
            
            if not is_similar and len(key_words) > 2:
                unique_sentences.append(sent)
                seen_content.append(key_words)  # Add set to list
        
        text = '. '.join(unique_sentences)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s+', r'\1 ', text)
        
        # Remove excessive emphasis markers
        text = re.sub(r'(\s+-\s+[^-]+\s+-\s+){2,}', ' ', text)
        
        # Ensure proper ending
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        return text
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        if not self.similarity_model:
            return 0.8  # Default similarity if model unavailable
            
        embeddings = self.similarity_model.encode([text1, text2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)
