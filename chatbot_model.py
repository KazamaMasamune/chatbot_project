import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib
import random
import os
import json
import sys
from datetime import datetime
import shutil

# Print the current working directory for debugging
print("Current working directory:", os.getcwd())

# Download NLTK data (only need to do this once)
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Our dataset: messages and their intents
training_data = [
    ("hi", "greet"),
    ("hello", "greet"),
    ("hey there", "greet"),
    ("good morning", "greet"),
    ("what's the weather like", "ask_weather"),
    ("how's the weather today", "ask_weather"),
    ("is it sunny in Florida", "ask_weather"),
    ("tell me the weather in New York", "ask_weather"),
    ("tell me a joke", "tell_joke"),
    ("make me laugh", "tell_joke"),
    ("say something funny", "tell_joke"),
    ("got any jokes", "tell_joke"),
]

# Split the data into messages (X) and intents (y)
messages, intents = zip(*training_data)
print("Messages:", messages)
print("Intents:", intents)

# Clean up the messages (remove "the", "is", etc., and make lowercase)
stop_words = set(stopwords.words("english"))

def clean_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(tokens)

# Clean all the messages
cleaned_messages = [clean_text(message) for message in messages]
print("Cleaned Messages:", cleaned_messages)

# Use absolute paths for the files in ~/Documents
documents_path = os.path.expanduser("~/Documents")
print(f"Documents path resolved to: {documents_path}")
model_filename = os.path.join(documents_path, "chatbot_model.pkl")
responses_filename = os.path.join(documents_path, "responses.json")
history_filename = os.path.join(documents_path, "chat_history.txt")

# Load or train the model
if os.path.exists(model_filename):
    print("Loading existing model...")
    model = joblib.load(model_filename)
else:
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(cleaned_messages, intents)
    print("Model trained! Your robot is ready to chat like a pro!")
    try:
        joblib.dump(model, model_filename)
        print(f"Model saved to {model_filename}!")
        if os.path.exists(model_filename):
            print(f"Confirmed: {model_filename} exists.")
        else:
            print(f"Error: {model_filename} was not created.")
            sys.exit(1)
    except Exception as e:
        print(f"Error saving model to {model_filename}: {e}")
        sys.exit(1)

# Define the default responses dictionary
default_responses = {
    'greet': [
        "Hey there, boss! What’s up—ready to have some fun?",
        "Hi! How’s my favorite user doing today?",
        "Hello! What can I do for you, rey?",
        "Arey, good to see you! What’s cooking, bro?"
    ],
    'ask_weather': [
        "Arey, it’s sunny in Florida—perfect for a picnic, bro!",
        "Let me check... Hmm, I think it’s sunny somewhere!",
        "Weather’s looking good—maybe go for a walk?",
        "I’m not a weather app, but I’d say it’s probably nice out there!"
    ],
    'tell_joke': [
        "Haha, here’s a good one: Why did the chicken join a band? Because it had the drumsticks!",
        "Why did the scarecrow become a motivational speaker? Because he was outstanding in his field!",
        "What do you call fake spaghetti? An impasta!",
        "Why don’t skeletons fight in school? They don’t have the guts for it!"
    ],
    'default': [
        "I’m not sure what you mean, rey! Can you say that again?",
        "Hmm, I didn’t catch that. Try something else!",
        "Arey, you’re confusing me—let’s try again!",
        "Boss, I’m a bit lost here—can you help me out?"
    ]
}

# Load or define the responses dictionary
try:
    if os.path.exists(responses_filename):
        print("Loading responses from file...")
        # Create a backup before loading
        backup_filename = responses_filename + ".backup"
        shutil.copy2(responses_filename, backup_filename)
        print(f"Created backup: {backup_filename}")
        with open(responses_filename, "r") as f:
            responses = json.load(f)
        print(f"Successfully loaded responses from {responses_filename}.")
    else:
        print("Responses file not found. Creating new responses file...")
        responses = default_responses
        with open(responses_filename, "w") as f:
            json.dump(responses, f, indent=4)
        if os.path.exists(responses_filename):
            print(f"Responses saved to {responses_filename}!")
        else:
            print(f"Error: {responses_filename} was not created.")
            sys.exit(1)
except Exception as e:
    print(f"Error handling responses file: {e}")
    responses = default_responses  # Fallback to default responses

# Function to predict the intent of a new message
def predict_intent(message):
    cleaned_message = clean_text(message)
    intent = model.predict([cleaned_message])[0]
    return intent

# Function to get a response based on the intent
def get_response(intent):
    return random.choice(responses.get(intent, responses['default']))

# Test the model with some new messages
test_messages = [
    "hello there",
    "what's the weather in Texas",
    "tell me a joke",
    "hi boss",
]

print("\nLet’s have some fun with the robot!")
for message in test_messages:
    intent = predict_intent(message)
    response = get_response(intent)
    print(f"User says: '{message}'")
    print(f"Robot guesses: '{intent}' intent")
    print(f"Robot says: {response}")
    print("-" * 50)

# Interactive chat loop with history saving
print("\nChat with the robot! Type 'exit' to stop.")
try:
    # Create a backup of chat_history.txt if it exists
    if os.path.exists(history_filename):
        backup_filename = history_filename + ".backup"
        shutil.copy2(history_filename, backup_filename)
        print(f"Created backup: {backup_filename}")
    with open(history_filename, "a") as history_file:
        print(f"Opened {history_filename} for writing.")
        if os.path.exists(history_filename):
            print(f"Confirmed: {history_filename} exists.")
        else:
            print(f"Error: {history_filename} was not created.")
            sys.exit(1)
        while True:
            try:
                user_input = input("You: ")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if user_input.lower().strip() == 'exit':
                    print("Robot: Goodbye, rey! See you next time!")
                    history_file.write(f"[{timestamp}] You: exit\n")
                    history_file.write(f"[{timestamp}] Robot: Goodbye, rey! See you next time!\n")
                    history_file.write("-" * 50 + "\n")
                    break
                
                cleaned_input = clean_text(user_input)
                intent = predict_intent(cleaned_input)
                response = get_response(intent)
                
                print(f"Robot guesses: '{intent}' intent")
                print(f"Robot says: {response}")
                print("-" * 50)
                
                history_file.write(f"[{timestamp}] You: {user_input}\n")
                history_file.write(f"[{timestamp}] Robot guesses: '{intent}' intent\n")
                history_file.write(f"[{timestamp}] Robot says: {response}\n")
                history_file.write("-" * 50 + "\n")
                history_file.flush()  # Ensure the file is written immediately
            except KeyboardInterrupt:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print("\nRobot: Caught a Ctrl+C! Goodbye, rey! See you next time!")
                history_file.write(f"[{timestamp}] You: [Interrupted with Ctrl+C]\n")
                history_file.write(f"[{timestamp}] Robot: Goodbye, rey! See you next time!\n")
                history_file.write("-" * 50 + "\n")
                break
except Exception as e:
    print(f"Error writing to chat history: {e}")
    sys.exit(1)