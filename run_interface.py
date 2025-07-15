#!/usr/bin/env python3
"""
Launch script for AI Humanizer Gradio Interface
"""

import os
import sys
from app import create_interface

def main():
    """Launch the Gradio interface"""
    print("🚀 Starting AI Humanizer Pro Interface...")
    print("📍 Interface will be available at: http://localhost:7860")
    print("🌐 Public link will be generated for sharing")
    print("⏹️  Press Ctrl+C to stop the server")
    
    interface = create_interface()
    
    try:
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,  # Creates public link
            show_error=True,
            favicon_path=None,
            show_api=True,
            debug=False
        )
    except KeyboardInterrupt:
        print("\n👋 Interface stopped. Thank you for using AI Humanizer Pro!")
    except Exception as e:
        print(f"❌ Error starting interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
