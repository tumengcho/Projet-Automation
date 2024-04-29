import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import random
import pyttsx3
import json
from googleapiclient.discovery import build
from googlesearch import search
import webbrowser
from rake_nltk import Rake
from sumy.parsers.html import HtmlParser
from sumy.nlp.tokenizers import Tokenizer as SumTok
from sumy.summarizers.lsa import LsaSummarizer
import time
import pyautogui
import ctypes
import spacy
import sys
from PyQt5.QtWidgets import QApplication, QLabel, QDesktopWidget
from PyQt5.QtCore import QTimer, Qt, pyqtSignal
from PyQt5.QtGui import QMovie
import threading

# Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')

# Load intents from JSON file
with open('Intents/Intent.json', 'r') as f:
    data = json.load(f)

# Clean text data
def clean(line):
    return ' '.join([word for word in line.split() if word.isalpha()])

# Initialize lists and dictionaries for intents
intents = []
unique_intents = []
text_input = []
response_for_intent = {}
function_for_intent = {}

# Extract data from JSON
for intent in data['intents']:
    if intent['intent'] not in unique_intents:
        unique_intents.append(intent['intent'])
    for text in intent['text']:
        text_input.append(clean(text))
        intents.append(intent['intent'])
    if intent['intent'] not in response_for_intent:
        response_for_intent[intent['intent']] = []
    for response in intent['responses']:
        response_for_intent[intent['intent']].append(response)
    function_for_intent[intent['intent']] = intent['extension']['function']

# Tokenize and pad sequences
tokenizer = Tokenizer(filters='', oov_token='<unk>')
tokenizer.fit_on_texts(text_input)
sequences = tokenizer.texts_to_sequences(text_input)
padded_sequences = pad_sequences(sequences, padding='pre')

# Convert intents to categorical vectors
intent_to_index = {intent: index for index, intent in enumerate(unique_intents)}
categorical_target = [intent_to_index[intent] for intent in intents]
num_classes = len(unique_intents)
categorical_vec = tf.keras.utils.to_categorical(categorical_target, num_classes=num_classes)

# Define and compile the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index) + 1, 300),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, dropout=0.1)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(padded_sequences, categorical_vec, epochs=200, verbose=0)
print(num_classes)

# Test data
test_text_inputs = ["Shutdown the computer",
                    "Open Google Chrome",
                    "Search the web for Python tutorials",
                    "Open gitkraken",
                    "Play muhammed hijab on youtube"
                    ]

test_intents = ["ShutdownComputer",
                "OpenApplication",
                "SearchWeb",
                "OpenApplication",
                "PlayYoutubeVideo"]

test_sequences = tokenizer.texts_to_sequences(test_text_inputs)
test_padded_sequences = pad_sequences(test_sequences, padding='pre')
test_labels = np.array([intent_to_index[intent] for intent in test_intents])
test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)
loss, accuracy = model.evaluate(test_padded_sequences, test_labels)

# Extract keywords from user input
def speak(text):
    # Initialize the text-to-speech engine
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
def extract_keywords(user_input):
    r = Rake()
    r.extract_keywords_from_text(user_input)
    return r.get_ranked_phrases()

# Generate response based on user input
def response(sentence):
    sent_tokens = [tokenizer.word_index.get(word, tokenizer.word_index['<unk>']) for word in sentence.split()]
    sent_tokens = tf.expand_dims(sent_tokens, 0)
    pred = model(sent_tokens)
    pred_class = np.argmax(pred.numpy(), axis=1)
    intent = unique_intents[pred_class[0]]
    return random.choice(response_for_intent[intent]), intent
def intent_has_entities(intent_name):
    for intent in data['intents']:
        if intent['intent'] == intent_name:
            return intent["extension"]["entities"]
    return False

# Search the web based on a query
def search_web(query):
    print(f"Searching the web for information about {query}...")
    webbrowser.open("https://www.google.com/search?q=" + query)
    results = list(search(query, num_results=10, lang="en", advanced=True))
    if not results:
        print("No results found.")
        return

    for i, result in enumerate(results, start=1):
        print(f" {result.title}")

    choice = input("Enter the number of the result you want to open (or 'q' to quit): ")
    if choice.lower() == 'q':
        print("Exiting search.")
        return
    try:

        choice_index = int(choice) - 1
        selected_result = results[choice_index]
        print(f"Opening: {selected_result.url}")
        webbrowser.open(selected_result.url)
    except (ValueError, IndexError):
        print("Invalid input or choice. Exiting search.")

def shutdown_computer():
    ctypes.windll.user32.LockWorkStation()

# Open an application
def open_application(app_name):
    time.sleep(2)

    # Press the Windows key to open the Start menu
    pyautogui.press('win')

    # Type the name of the application
    pyautogui.write(app_name, interval=0.2)
    # Press Enter to open the application
    pyautogui.press('enter')
    print(f"{app_name} has been opened!")
def play_youtube(query):
    api_key = 'AIzaSyAHS1FjoX45zJcvAA1vv8X1G4A7o5H7waA'

    try:
        # Build the YouTube API service
        youtube = build('youtube', 'v3', developerKey=api_key)

        # Call the search.list method to search for videos
        search_response = youtube.search().list(
            q=query,
            part='snippet',
            type='video',
            maxResults=1
        ).execute()

        # Extract the video ID from the search results
        video_id = search_response['items'][0]['id']['videoId']

        # Construct the URL for the video
        video_url = f'https://www.youtube.com/watch?v={video_id}'

        # Open the video URL in the default web browser
        webbrowser.open(video_url)
        time.sleep(3)
        pyautogui.press("f")
    except Exception as e:
        print(f"An error occurred: {e}")
def extract_entities(text, model_path="ner_model"):
        # Load the trained NER model
    nlp = spacy.load(model_path)

        # Process the input text
    doc = nlp(text.lower())

        # Extract entities from the processed document
    entities_info = [(ent.label_, ent.text) for ent in doc.ents]

    return entities_info

def summarize_webpage(url, num_sentences=4):
    # Parse HTML content of the webpage
    parser = HtmlParser.from_url(url, SumTok("english"))

    # Initialize LSA summarizer
    summarizer = LsaSummarizer()

    # Summarize the parsed content
    summary = summarizer(parser.document, num_sentences)

    # Return the summarized text
    return " ".join([str(sentence) for sentence in summary])


# Perform actions based on detected intent
def JarvisAi(query):
    bot_response, intent = response(query)

    if intent_has_entities(intent):

        entities = extract_entities(query)
        print(entities)
        if entities:
            jarvis_reponse, value = replace_placeholders(bot_response, entities)
            print(entities)
            if value:
                print("first")
                speak(jarvis_reponse)
                function_name = function_for_intent[intent]
                globals()[function_name](value)
        else:
            speak("{} . Intent is {} -- but value is none existent. Please teach me.".format(query, intent))
    elif intent in function_for_intent:
        # Call the function corresponding to the detected intent
        speak(bot_response)
        function_name = function_for_intent[intent]
        globals()[function_name]()


def replace_placeholders(response, entities):
    """
    Replace placeholders in a response with actual entity values.

    Args:
    - response (str): The response string containing placeholders.
    - entities (list of tuples): List of tuples containing entity labels and values.

    Returns:
    - str: The response string with placeholders replaced by actual values.
    """
    if entities is None:
        return response, None

    replaced_response = response
    for label, value in entities:
        # Convert label to uppercase and surround with square brackets
        placeholder = f"[{label.upper()}]"
        # Replace placeholder with actual value
        replaced_response = replaced_response.replace(placeholder, value)
        print(replaced_response)
        if label and value:
            return replaced_response, value



class JarvisApp(QLabel):
    image_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.movie_path = "images/listening.gif"
        self.init_ui()

    def init_ui(self):
        self.set_movie(self.movie_path)
        self.setAlignment(Qt.AlignCenter)
        self.setScaledContents(True)  # Scale the movie to fit the label
        # Start a timer to change the movie every 2 seconds
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.change_movie)
        self.timer.start(2000)

    def set_movie(self, movie_path):
        self.movie_path = movie_path
        self.movie = QMovie(self.movie_path)
        self.setMovie(self.movie)
        self.movie.start()

    def change_movie(self):
        # Change the movie dynamically based on user input or other conditions
        pass

    def update_movie(self, movie_path):
        self.set_movie(movie_path)

def main_loop(jarvis_app):
    while True:
        jarvis_app.image_changed.emit("images/speaking.gif")
        query = input('You: ')
        if query.lower() == 'quit':
            break
        jarvis_app.image_changed.emit("images/listening.gif")
        JarvisAi(query)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    jarvis_app = JarvisApp()
    jarvis_app.setWindowFlags(Qt.FramelessWindowHint)  # Remove window frame
    jarvis_app.setAttribute(Qt.WA_TranslucentBackground)  # Make window background transparent

    # Get the screen resolution
    desktop = QDesktopWidget()
    screen_rect = desktop.screenGeometry()
    jarvis_app.setGeometry(screen_rect)

    jarvis_app.show()

    # Connect the image_changed signal to the update_movie slot
    jarvis_app.image_changed.connect(jarvis_app.update_movie)

    # Start the Jarvis main loop in a separate thread
    thread = threading.Thread(target=main_loop, args=(jarvis_app,))
    thread.start()

    sys.exit(app.exec_())
