#!/usr/bin/env python3
"""
AI Humanizer - Main Entry Point
Professional AI text humanization tool with advanced features.
"""

import time
from src.humanizer import SuperiorHumanizer
from src.config import Config
from src.utils.logger import setup_logger

def main():
    """Main function to run the AI Humanizer"""
    # Setup logging
    logger = setup_logger()
    logger.info("Starting AI Humanizer...")
    
    # Initialize the humanizer
    print("Initializing Superior AI Humanizer...")
    humanizer = SuperiorHumanizer()
    print("✅ Superior Humanizer ready with advanced features!")
    
    # Test text (you can replace this with input from file or user)
    test_text = """
    Dogs are among the most loyal and loving animals on Earth. Known as "man's best friend," dogs have been companions to humans for thousands of years. They offer unconditional love, protection, and joy to their owners. Dogs come in various breeds, sizes, and temperaments, making them suitable for different families and lifestyles.
    One of the most admirable qualities of dogs is their loyalty. They form strong bonds with their owners and are always eager to please. Dogs are intelligent animals and can be trained to perform various tasks—from simple tricks to complex roles such as search and rescue or assisting people with disabilities.
    Beyond companionship, dogs have a calming effect on humans. Many studies show that having a dog can reduce stress, encourage physical activity, and improve mental health. Children who grow up with dogs often learn responsibility, empathy, and patience.
    Dogs also serve society in many important ways. Police dogs help in crime detection, therapy dogs provide comfort in hospitals, and guide dogs assist the visually impaired.
    In conclusion, dogs are more than just pets—they are family. Their loyalty, love, and usefulness make them one of the most cherished animals in the world. Everyone can benefit from the joy a dog brings.
    """
    
    print("Starting enhanced humanization process...\n")
    
    start_time = time.time()
    humanized_text = humanizer.advanced_humanization(
        test_text, 
        style="casual", 
        persona="thoughtful",
        preserve_threshold=0.8
    )
    
    elapsed_time = time.time() - start_time
    print(f"\nHumanization completed in {elapsed_time:.2f} seconds")
    
    print("\n" + "="*80)
    print("ORIGINAL TEXT:")
    print("="*80)
    print(test_text)
    
    print("\n" + "="*80)
    print("ENHANCED HUMANIZED TEXT:")
    print("="*80)
    print(humanized_text)

if __name__ == "__main__":
    main()
