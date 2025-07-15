# 🧠 AI Humanizer – Make AI Text Sound More Human

Hi, I'm Prashant 👋 This is a project I built to solve a real problem: most AI-generated text sounds… well, like it was written by a machine. So I created **AI Humanizer** — a tool that rewrites AI text to make it sound more natural, relatable, and human, while still keeping its original meaning.

## 🚀 What It Does

AI Humanizer rewrites stiff, robotic content and makes it feel alive — like something a real person would say. Here's what it offers:

- ✍️ **Paraphrasing with Multiple Models**: Supports T5, BART, and Google's Gemini API to create diverse, human-like rewordings.
- 🎨 **Style Customization**: Switch between casual, conversational, and professional tones.
- 👤 **Persona Adaption**: Want your content to sound more enthusiastic or analytical? Just pick a persona.
- 🧠 **Topic Awareness**: Automatically adjusts based on what you're writing about.
- ✅ **Grammar Check**: Built-in grammar and typo correction.
- 🔍 **AI Detection Reduction**: Lowers your content's "AI score" to help it bypass AI detectors.
- 🔒 **Meaning Preservation**: Keeps your original message intact.

## ⚙️ How to Get Started

Clone the repo and set things up:

```bash
git clone <your-repo-url>
cd aihumanizer_final
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the spaCy model (needed for text analysis):

```bash
python -m spacy download en_core_web_sm
```

### Optional Setup

- **For grammar checking**: Install Java and make sure it's in your PATH.
- **For Gemini API**: Add your key in a `.env` file like this:

```bash
GEMINI_API_KEY=your_key_here
```

## ▶️ How to Use

**Run via Terminal:**

```bash
python main.py
```

**Or Use the Gradio Web UI:**

```bash
python app.py
```

You'll get an interactive interface where you can paste your AI content and tweak how human you want it to sound.

## 🔍 Sample Output
<img width="1715" height="1072" alt="image" src="https://github.com/user-attachments/assets/64adbe43-ce95-4ebc-8ea6-365c05278f57" />


## 🛠 Project Layout

```
aihumanizer_final/
├── main.py                 # Core runner
├── app.py                  # Web interface (Gradio)
├── src/
│   ├── config.py          # Settings and keys
│   ├── humanizer.py       # Main logic
│   ├── paraphrasing/      # Text rewriters
│   ├── patterns/          # Rule-based pattern tools
│   ├── models/            # Model loader
│   └── utils/             # Logger and helpers
├── requirements.txt
└── README.md
```

## 🧪 Troubleshooting

- ❗ **spaCy Errors?** → Make sure you ran the `download` command for `en_core_web_sm`.
- ❗ **Grammar Checking Fails?** → Double check if Java is installed.
- ❗ **Gemini Not Working?** → Recheck your API key.

## ✏️ Customization

You can tweak `src/config.py` for:
- Model weights and options
- Gemini API settings
- Persona presets
- Logging preferences

## 🧑‍💻 About Me

I'm Prashant, an AI/ML engineer who loves building practical tools that make AI more useful and user-friendly. This project was born out of a real need — I was tired of AI text sounding lifeless and obvious.

If this tool helps you or if you have ideas to improve it, I'd love to hear from you!

## 📄 License

This project is **not open source**.  
All content in this repository is © 2025 Prashant Singh. All rights reserved.

You're free to explore the code, but reuse or redistribution is **not allowed** without my written permission.

Feel free to reach out if you'd like to collaborate or license the project.
