"""Professional Gradio Interface for AI Humanizer"""
import gradio as gr
import time
import sys
import os
from typing import Dict, Tuple, List

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the AI Humanizer components
try:
    from src.humanizer import SuperiorHumanizer
    from src.config import Config
    from src.utils.logger import setup_logger
except ImportError as e:
    print(f"Error importing AI Humanizer components: {e}")
    print("Please ensure all required files are in the correct location.")
    sys.exit(1)

class GradioInterface:
    """Professional Gradio interface for AI Humanizer"""
    
    def __init__(self):
        """Initialize the Gradio interface"""
        print("🚀 Initializing AI Humanizer Gradio Interface...")
        try:
            self.logger = setup_logger()
            self.humanizer = None  # Lazy loading
            self.config = Config()
            print("✅ AI Humanizer interface initialized successfully!")
        except Exception as e:
            print(f"❌ Error initializing AI Humanizer: {e}")
            raise
    
    def initialize_humanizer(self):
        """Initialize the humanizer (lazy loading)"""
        if self.humanizer is None:
            self.logger.info("Loading AI Humanizer models...")
            self.humanizer = SuperiorHumanizer()
            self.logger.info("✅ AI Humanizer loaded successfully!")
        return self.humanizer
    
    def humanize_text_interface(self, 
                               text: str, 
                               style: str, 
                               persona: str, 
                               preserve_threshold: float,
                               topic_override: str,
                               progress=gr.Progress()) -> Tuple[str, str]:
        """
        Main humanization interface function
        
        Returns:
            Tuple of (humanized_text, status_message)
        """
        if not text or not text.strip():
            return "", "❌ Please enter some text to humanize."
        
        if len(text.strip()) < 10:
            return "", "❌ Text too short. Please enter at least 10 characters."
        
        try:
            progress(0.1, desc="Initializing models...")
            humanizer = self.initialize_humanizer()
            
            progress(0.3, desc="Starting humanization process...")
            
            # Humanize the text
            humanized_text = humanizer.advanced_humanization(
                text,
                style=style.lower(),
                persona=persona.lower(),
                preserve_threshold=preserve_threshold,
                topic=topic_override if topic_override != "Auto-detect" else None
            )
            
            progress(1.0, desc="Complete!")
            
            # Simple success status
            status_msg = f"✅ Text successfully humanized!"
            
            return humanized_text, status_msg
            
        except Exception as e:
            error_msg = f"❌ Error during humanization: {str(e)}"
            self.logger.error(f"Humanization error: {e}")
            return "", error_msg
    
    def load_sample_text(self, sample_type: str) -> str:
        """Load sample texts"""
        samples = {
            "Academic": """Artificial intelligence represents a paradigm shift in computational methodologies, fundamentally altering the landscape of technological innovation. The implementation of machine learning algorithms facilitates enhanced decision-making processes, thereby optimizing operational efficiency across numerous domains. Furthermore, the integration of neural networks demonstrates significant potential for advancing predictive analytics capabilities.""",
            
            "Business": """Our comprehensive analysis indicates substantial growth opportunities within the emerging market segments. The strategic implementation of innovative solutions will facilitate enhanced customer engagement metrics. Furthermore, operational optimization initiatives demonstrate significant potential for maximizing return on investment while maintaining competitive advantages in the marketplace.""",
            
            "Creative": """The magnificent sunset painted the sky in brilliant hues of orange and crimson, creating a breathtaking spectacle that captivated all who witnessed it. The gentle breeze carried the sweet fragrance of blooming flowers across the meadow, where children played joyfully among the tall grass. This picturesque scene embodied the perfect harmony between nature's beauty and human happiness.""",
            
            "Technical": """The system architecture implements a microservices-based approach, utilizing containerized deployment strategies to ensure scalability and maintainability. The application programming interface facilitates seamless integration between disparate systems, while the database optimization protocols enhance query performance and data retrieval efficiency. Additionally, the implementation of caching mechanisms significantly reduces latency."""
        }
        return samples.get(sample_type, "")
    
    def create_interface(self) -> gr.Blocks:
        """Create the Gradio interface"""
        
        # Custom CSS for better styling
        custom_css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .status-success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .status-error {
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
        """
        
        with gr.Blocks(
            title="AI Humanizer Pro",
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="gray",
                neutral_hue="gray"
            ),
            css=custom_css
        ) as interface:
            
            # Header
            gr.Markdown("""
            <div class="main-header">
                <h1>🤖➡️👤 AI Humanizer Pro</h1>
                <p>Transform AI-generated text into natural, human-like content with advanced controls</p>
            </div>
            """)
            
            # Main Interface
            with gr.Tabs():
                
                # Single Text Processing Tab
                with gr.TabItem("📝 Text Humanization", elem_id="single-tab"):
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 📝 Input Text")
                            input_text = gr.Textbox(
                                label="Text to Humanize",
                                placeholder="Paste your AI-generated text here...",
                                lines=12,
                                max_lines=20,
                                info="Enter the text you want to humanize (minimum 10 characters)"
                            )
                            
                            with gr.Row():
                                sample_dropdown = gr.Dropdown(
                                    choices=["Academic", "Business", "Creative", "Technical"],
                                    label="📚 Load Sample Text",
                                    value=None
                                )
                                load_sample_btn = gr.Button("Load Sample", size="sm")
                            
                            with gr.Accordion("⚙️ Advanced Settings", open=False):
                                with gr.Row():
                                    style_dropdown = gr.Dropdown(
                                        label="🎨 Writing Style",
                                        choices=["casual", "conversational", "professional"],
                                        value="casual",
                                        info="Choose the desired writing style"
                                    )
                                    
                                    persona_dropdown = gr.Dropdown(
                                        label="👤 Persona",
                                        choices=["casual", "thoughtful", "enthusiastic", "analytical"],
                                        value="thoughtful",
                                        info="Select the personality tone"
                                    )
                                
                                with gr.Row():
                                    preserve_slider = gr.Slider(
                                        label="🎯 Content Preservation",
                                        minimum=0.7,
                                        maximum=0.95,
                                        value=0.8,
                                        step=0.05,
                                        info="Higher values preserve more original meaning"
                                    )
                                    
                                    topic_override = gr.Dropdown(
                                        choices=["Auto-detect", "technology", "health", "education", "general"],
                                        value="Auto-detect",
                                        label="🏷️ Topic Override"
                                    )
                            
                            humanize_btn = gr.Button(
                                "🚀 Humanize Text",
                                variant="primary",
                                size="lg"
                            )
                        
                        with gr.Column(scale=1):
                            gr.Markdown("### ✨ Humanized Output")
                            output_text = gr.Textbox(
                                label="Humanized Text",
                                lines=12,
                                max_lines=20,
                                show_copy_button=True,
                                info="Your humanized text will appear here"
                            )
                            
                            status_message = gr.Markdown(
                                value="Ready to humanize your text!",
                                elem_classes=["status-success"]
                            )
                
                # About Tab
                with gr.TabItem("ℹ️ About & Help", elem_id="about-tab"):
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("""
                            ### 📖 How to Use
                            
                            1. **Input Text**: Paste your AI-generated text in the input box
                            2. **Choose Settings**: Select writing style, persona, and other preferences  
                            3. **Humanize**: Click the "Humanize Text" button
                            4. **Copy**: Use the copy button to get your humanized text
                            
                            ### 🎛️ Parameter Guide
                            
                            - **Writing Style**: Controls formality level
                              - *Casual*: Relaxed, everyday language
                              - *Conversational*: Natural dialogue style
                              - *Professional*: Formal business tone
                            
                            - **Persona**: Adjusts personality traits
                              - *Casual*: Laid-back approach
                              - *Thoughtful*: Reflective and considered
                              - *Enthusiastic*: Energetic and positive
                              - *Analytical*: Data-driven and logical
                            
                            - **Content Preservation**: Higher values maintain more original meaning
                            """)
                        
                        with gr.Column():
                            gr.Markdown("""
                            ### 💡 Tips for Best Results
                            
                            - Use longer texts (50+ words) for better results
                            - Adjust preservation threshold based on your needs
                            - Try different style/persona combinations
                            - For technical content, use "professional" style
                            - For blog posts, "conversational" works well
                            
                            ### ✨ Features
                            - Multi-model paraphrasing with 5+ AI models
                            - Context-aware processing
                            - Grammar checking and correction
                            - Topic detection and adaptation
                            - Real-time processing with progress tracking
                            
                            ### 🎯 Writing Styles
                            
                            **Casual**
                            - Relaxed, everyday language
                            - Shorter sentences
                            - Informal tone
                            
                            **Conversational**
                            - Natural dialogue style
                            - Engaging and friendly
                            - Personal touch
                            
                            **Professional**
                            - Formal business tone
                            - Clear and precise
                            - Structured approach
                            """)
            
            # Event Handlers
            load_sample_btn.click(
                fn=self.load_sample_text,
                inputs=[sample_dropdown],
                outputs=[input_text]
            )
            
            humanize_btn.click(
                fn=self.humanize_text_interface,
                inputs=[
                    input_text, style_dropdown, persona_dropdown, 
                    preserve_slider, topic_override
                ],
                outputs=[output_text, status_message],
                show_progress=True
            )
            
            # Footer
            gr.Markdown("""
            <div style="text-align: center; margin-top: 2rem; padding: 1rem; border-top: 1px solid #eee;">
                <p>🤖 AI Humanizer Pro v1.0 | Built with ❤️ using Gradio | Transform AI text into natural, human-like content</p>
            </div>
            """)
        
        return interface
    
    def launch(self, **kwargs):
        """Launch the Gradio interface"""
        interface = self.create_interface()
        
        # Default launch parameters
        launch_params = {
            "server_name": "0.0.0.0",
            "server_port": 7860,
            "share": False,
            "debug": False,
            "show_error": True,
            "quiet": False
        }
        
        # Update with user parameters
        launch_params.update(kwargs)
        
        print(f"""
🚀 AI Humanizer Pro Interface Starting...
🌐 Local URL: http://localhost:{launch_params['server_port']}
📝 Ready to humanize your text!
        """)
        
        interface.launch(**launch_params)

def main():
    """Main function to run the Gradio interface"""
    try:
        # Create and launch the interface
        gradio_app = GradioInterface()
        
        gradio_app.launch(
            # share=True,   # Set to True to create a public link
            debug=False,  # Set to True for development  
            show_error=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 Interface stopped by user")
    except Exception as e:
        print(f"❌ Error launching interface: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()