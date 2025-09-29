# 📝 AI Companion – Fine-Tuned Creative Assistant

## 🚀 Project Overview

This project fine-tunes a **large language model** using **PEFT (Parameter-Efficient Fine-Tuning)** with **LoRA (Low-Rank Adaptation)** and integrates it into a **Streamlit app**.
The app acts as a **creative assistant** to help users generate **stories, recipes, and poetry**, optimized to adapt to **user moods**.

✨ **Key highlights**:

* Fine-tuned model with PEFT + LoRA
* Emotion detection
* Text-to-speech conversion
* Streamlit UI for seamless interaction

---

## 🔑 Key Features

* **Fine-Tuning with PEFT + LoRA**: Efficiently fine-tunes *NousResearch/Llama-2-7b-chat-hf*.
* **Creative Content Generation**: Stories, recipes, and poetry generated from user prompts.
* **Emotion Detection**: Tailored content based on detected mood.
* **Text-to-Speech**: Converts text output into audio with GTTS.
* **Streamlit Integration**: Clean, interactive, and user-friendly interface.

---

## 📂 Project Structure

```bash
├── ai-companion.py                 # Streamlit app
├── fine_tuning.py                  # Fine-tuning script
├── datasets/
│   ├── stories_data.csv
│   ├── recipes_data.csv
│   ├── poetry_data.csv
├── models/
│   ├── fine_tuned_model_LLAMA-2/   # Fine-tuned model files
├── emotion_detection_model/
│   ├── DistilBert                  # Emotion classifier
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation
```

---

## 📝 Key Scripts

* **fine_tuning.py** – fine-tunes the model using PEFT + LoRA.
* **ai-companion.py** – loads the pre-trained + fine-tuned model, integrates with Streamlit, generates content, detects emotions, and handles TTS.

---

## ⚙️ Installation & Setup

### 1️⃣ Install Dependencies

You can use pip directly:

```bash
pip install datasets transformers peft tensorflow langchain langchain-community streamlit gtts pyngrok
```

or install from the requirements file:

```bash
pip install -r requirements.txt
```

### 2️⃣ Add Fine-Tuned Model

Place your saved LoRA adapter files:

```
adaptor_config.json  
adaptor_model.safetensors  
```

inside:

```
models/fine_tuned_model_LLAMA-2/
```

### 3️⃣ Run the Streamlit App

```bash
streamlit run ai-companion.py
```

---

## 🖥 Streamlit App Features

* **User Interaction** – Input a prompt to generate a story, recipe, or poem.
* **Emotion Detection** – Uses DistilBERT classifier to adapt responses.
* **Response Generation** – Tailored text output based on input + detected emotion.
* **Text-to-Speech** – Play generated content as audio.

---

## 🎯 Usage

1. Enter your prompt (story/recipe/poem).
2. Click **Generate Response**.
3. Listen to the generated audio.

**Example Prompts:**

* Story: *"Tell me a story about a brave knight."*
* Recipe: *"Give me a recipe for chocolate cake."*
* Poem: *"Write a poem about love."*

---

## 🧠 Emotion Classifier

* Built using **Hugging Face DistilBERT** model.
* Detects moods like: Joy, Desire, Admiration, Approval, Curiosity, Fear, Sadness, Anger, and Neutral.

---

## 🔊 Text-to-Speech

* Uses **GTTS (Google Text-to-Speech)**.
* Audio playback inside the Streamlit app.

---

## 👥 Authors & Contributors

**Author:**

* Wasif Mehboob

**Contributors:**

* Ahsan Waseem
* Abdul Moiz
* Ameer Hamza

---

## 🙏 Acknowledgments

* [Hugging Face](https://huggingface.co/) for model + libraries.
* [Streamlit](https://streamlit.io/) community for simplifying interactive apps.

---

Would you like me to **add badges** (e.g., Python version, Streamlit badge, Hugging Face badge) to make the README look even more professional? (I can do that next.)
