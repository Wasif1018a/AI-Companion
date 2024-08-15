Project Overview:
This project involves fine-tuning a language model using PEFT (Parameter-Efficient Fine-Tuning) with LoRA (Low-Rank Adaptation) and integrating it into a Streamlit app. The Streamlit app acts as a creative assistant, helping users generate stories, recipes, and poetry. The fine-tuned model is optimized for generating content based on user mood, and the app includes functionalities like emotion detection and text-to-speech conversion.

Key Features:
Fine-Tuning with PEFT and LoRA: Efficient fine-tuning of the "NousResearch/Llama-2-7b-chat-hf" model using LoRA.
Creative Content Generation: The app can generate stories, recipes, and poetry based on user input and detected mood.
Emotion Detection: Emotion classification to tailor responses.
Text-to-Speech: Converts generated responses to audio.
Streamlit Integration: A user-friendly interface with a custom design for interaction.

Project Structure:
bash
Copy code
├── streamlit_app_2.py          # Streamlit app code
├── fine_tuning.py              # Fine-tuning script
├── datasets/
│   ├── stories_data.csv        # Dataset with stories for fine-tuning
├── models/
│   ├── fine_tuned_model/       # Directory to save the fine-tuned model
├── utils/
│   ├── emotion_classifier.py   # Emotion classifier script
├── requirements.txt            # Required Python libraries
└── README.md                   # Project documentation

Key Scripts
streamlit_app.py: Main Streamlit app that interacts with users, generates content, predicts emotions, and converts text to audio.
fine_tuning.py: Python script to fine-tune the language model using PEFT and LoRA.
ai-companion.py: Contains code for loading the pre-trained model and applying the saved fine-tuned model configurations using PEFT and LoRA.

Installation and Setup:
Step 1: Install Required Libraries

Step 2: Fine-tune the Model
bash
Copy code
python fine_tuning.py

Step 3: Run the Streamlit App
bash
Copy code
streamlit run ai-companion.py

Fine-Tuning Process:
Loading Dataset: Load the dataset from a CSV file containing stories.
Tokenization: Tokenize the dataset using the AutoTokenizer from the Hugging Face library.
LoRA Configuration: Apply LoRA configuration to the model for parameter-efficient fine-tuning.
Training: Use the Trainer API from Hugging Face to train the model with the specified training arguments.

Streamlit App Features:
User Interaction: Users can input their prompts in the text field and receive generated content based on their mood.
Emotion Detection: The app detects the user's mood from the input text using an emotion classifier.
Response Generation: Based on the input and detected mood, the app generates a relevant story, recipe, or poem.
Text-to-Speech: The generated content is converted to an audio that users can listen to directly within the app.

Usage:
Enter a prompt: Provide a prompt related to a story, recipe, or poem.
Generate Response: Click the "Generate Response" button.
Listen to the Response: The response will be displayed and also can be played as audio.

Example Prompts:
Story: "Tell me a story about a brave knight."
Recipe: "Give me a recipe for chocolate cake."
Poem: "Write a poem about love."

Emotion Classifier:
The emotion classifier uses a pre-trained model from the Hugging Face library to detect the user's mood from the input text.
Supported emotions include Joy, Desire, Admiration, Approval, Curiosity, Fear, Sadness, Anger, and Neutral.

Text-to-Speech:
The app uses GTTS (Google Text-to-Speech) to convert the generated responses to audio.
The audio file is played within the app.

Authors:
Wasif Mehboob

Contributors:
Ahsan Waseem
Abdul Moiz
Ameer Hamza

Acknowledgments
Thanks to the Hugging Face community for the amazing libraries and models.
Special thanks to the Streamlit community for making interactive web apps easy to build.
