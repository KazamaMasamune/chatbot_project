Chatbot Project 🤖💬 :

A fun and simple chatbot built with Python, NLTK, and scikit-learn! 🎉 It can greet you, answer weather questions, and tell jokes using natural language processing and a Naive Bayes classifier. 😄
✨ Features

👋 Responds to greetings (e.g., "hi", "hello")
☀️ Answers weather-related questions (e.g., "what's the weather like")
😂 Tells jokes on request (e.g., "tell me a joke")
📜 Saves chat history to a file (chat_history.txt)
💾 Persists the trained model and responses for reuse

🛠️ Prerequisites

🐍 Python 3.6 or higher
🌐 Git (to clone the repository)

🚀 Setup

Clone the RepositoryClone this project from GitHub to get started! 📥
git clone https://github.com/your-username/chatbot_project.git
cd chatbot_project


Create a Virtual EnvironmentSet up a virtual environment to keep things tidy! 🧹
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install DependenciesInstall the required Python packages to power the chatbot! 📦
pip install nltk numpy scikit-learn joblib


Download NLTK DataThe chatbot uses NLTK for natural language processing. Download the required data to make it work! 🌍
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')



🎮 Usage

Run the ChatbotLaunch the chatbot and start chatting! 🚀
python3 chatbot_model.py


Interact with the Chatbot

The chatbot will prompt you to type a message. 💬
Examples of inputs:
"hi" → Greets you back 👋
"what's the weather like" → Gives a weather-related response ☀️
"tell me a joke" → Shares a joke 😂


Type exit to stop the chatbot. 👋


Chat History

Conversations are saved in chat_history.txt in the project directory. 📜
The trained model (chatbot_model.pkl) and responses (responses.json) are also saved for future use. 💾



📂 Project Structure

chatbot_model.py: The main script containing the chatbot logic. 🧠
.gitignore: Excludes venv/, chatbot_model.pkl, chat_history.txt, and backup files from Git. 🚫
chatbot_model.pkl: (Generated) The trained machine learning model. 📈
responses.json: (Generated) The responses dictionary in JSON format. 📝
chat_history.txt: (Generated) Logs of your conversations with the chatbot. 🗒️

💬 Example Interaction
Chat with the robot! Type 'exit' to stop. 🤗
You: hi
Robot guesses: 'greet' intent
Robot says: Hello! What can I do for you, rey? 👋
--------------------------------------------------
You: what's the weather in Texas
Robot guesses: 'ask_weather' intent
Robot says: Arey, it’s sunny in Florida—perfect for a picnic, bro! ☀️
--------------------------------------------------
You: tell me a joke
Robot guesses: 'tell_joke' intent
Robot says: What do you call fake spaghetti? An impasta! 😂
--------------------------------------------------
You: exit
Robot: Goodbye, rey! See you next time! 👋

📝 Notes

The chatbot uses a simple Naive Bayes classifier with TF-IDF vectorization for intent prediction. 🧠
Responses are randomly selected from a predefined set for each intent. 🎲
Generated files (chatbot_model.pkl, responses.json, chat_history.txt) are excluded from the repository via .gitignore. 🚫

📜 License
This project is licensed under the MIT License - see the LICENSE file for details (if you add one). 📄
👨‍💻 Author
Goutham Kaza
