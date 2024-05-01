import tkinter as tk
from tkinter import filedialog
import speech_recognition as sr
import threading
from keras.models import load_model
import cv2
from PIL import Image
from PIL import ImageTk
import pickle
import tkinter as tk
from tkinter import filedialog
import speech_recognition as sr
import threading
import os
# utilisée pour les calculs mathématiques en Python, spécialement pour les tableaux à plusieurs dimensions.
import numpy as np
#  convertir des données textuelles en séquences numériques qui peuvent être utilisées comme entrées pour les réseaux de neurones.
from keras.preprocessing.text import Tokenizer
# permet d'assurer que toutes les séquences de données ont la même longueur et peuvent donc être utilisées comme entrées pour le modèle,ajoutant des zéros à la fin des séquences plus courtes pour s'assurer que toutes les entrées ont la même longueur 
from tensorflow.keras.preprocessing.sequence import pad_sequences
# Sequential est une classe de Keras qui permet de définir et créer des modèles de réseaux de neurones séquentiels en permettant d'ajouter les couches du modèle les unes après les autres de manière simple et conviviale
from keras.models import Sequential
# import des couches Embedding[vecteur de densite latente], GRU et Dense(effectuer les calculs de regression) pour le modèle RNN
from keras.layers import Embedding, GRU, Dense 
# stocker et manipuler des données tabulaires, telles que des tableaux de données enregistrés dans des fichiers CSV
import pandas as pd
import keras
import pickle 
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from keras.models import load_model

# Charger le tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

# Charger le modèle sans réentraîner
loaded_model = load_model('sentiment_analysis_model.h5')

# Charger le modèle d'image sans entraîner
model_path = "model1.h5"
loaded_image_model = keras.models.load_model(model_path)

# Load the saved model of image sentiment
model = keras.models.load_model('happysadmodel.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    # Resize the image to 256x256 pixels
    image = image.resize((256, 256))
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Normalize the pixel values to be between 0 and 1
    image_array = image_array / 255.0
    # Add a batch dimension to the array
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Define a function to make a prediction using the model
def predict(image):
    # Preprocess the image
    image_array = preprocess_image(image)
    # Use the model to make a prediction
    prediction = model.predict(image_array)
    # Return the prediction
    return prediction

class InputApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Input Application")

        # Configure le style
        self.configure(background="#f0f0f0")  # Couleur de fond globale
        self.option_add('*TCombobox*Listbox.background', '#ffffff')  # Couleur de fond des listes déroulantes
        self.option_add('*TCombobox*Listbox.foreground', '#000000')  # Couleur du texte dans les listes déroulantes

        self.input_type = tk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        # Créer un cadre pour contenir les éléments
        frame = tk.Frame(self, bg="#ffffff", padx=20, pady=20)  # bg: couleur de fond, padx/pady: marge interne
        frame.pack(padx=20, pady=20)

        # Titre de l'application
        title_label = tk.Label(frame, text="Input Application", font=('Helvetica', 18, 'bold'), bg="#ffffff")
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        # Radio buttons pour la sélection d'entrée
        tk.Radiobutton(frame, text="Text", variable=self.input_type, value="text", bg="#ffffff").grid(row=1, column=0, sticky=tk.W)
        tk.Radiobutton(frame, text="Voice", variable=self.input_type, value="voice", bg="#ffffff").grid(row=2, column=0, sticky=tk.W)
        tk.Radiobutton(frame, text="Image", variable=self.input_type, value="image", bg="#ffffff").grid(row=3, column=0, sticky=tk.W)

        # Bouton Soumettre
        submit_button = tk.Button(frame, text="Submit", command=self.process_input)
        submit_button.grid(row=4, column=0, pady=(20, 0))

    def process_input(self):
        input_type = self.input_type.get()

        if input_type == "text":
            self.get_text_input()
        elif input_type == "voice":
            self.record_audio()
        elif input_type == "image":
            self.select_image()

    def get_text_input(self):
        text_window = tk.Toplevel(self)
        text_window.title("Text Input")

        tk.Label(text_window, text="Enter text:").pack()
        entry = tk.Entry(text_window)
        entry.pack()

        def submit_text():
            entered_text = entry.get()
            # Tokenisation et pad de la séquence
            sequence = loaded_tokenizer.texts_to_sequences([entered_text])
            # Puis ajoute des zéros pour faire en sorte que la longueur de la séquence soit de 100.
            padded_sequence = pad_sequences(sequence, maxlen=100)
            # Prédiction du sentiment
            prediction = loaded_model.predict(padded_sequence)[0][0]
            sentiment = "positive 😄" if prediction > 0.5 else "negative 😔"
            tk.Label(text_window, text=f"Your sentiment is: {sentiment}").pack()

        submit_button = tk.Button(text_window, text="Submit", command=submit_text)
        submit_button.pack()

    def record_audio(self):
        audio_window = tk.Toplevel(self)
        audio_window.title("Voice Input")

        recognizer = sr.Recognizer()

        def start_recording():
            def record_audio_thread():
                nonlocal recognizer  # Make recognizer accessible inside the thread
                with sr.Microphone() as source:
                    audio_window.title("Recording...")
                    try:
                        self.audio = recognizer.listen(source, timeout=60*60*24)  # Adjust timeout as needed
                    except sr.WaitTimeoutError:
                        audio_window.title("Stopped Recording")
                        return

                try:
                    transcribed_text = recognizer.recognize_google(self.audio)
                    sequence = loaded_tokenizer.texts_to_sequences([transcribed_text])
                    padded_sequence = pad_sequences(sequence, maxlen=100)
                    prediction = loaded_model.predict(padded_sequence)[0][0]
                    sentiment="For the sentence  {transcribed_text} the sentiment is :"
                    sentiment = "positive \U0001f600" if prediction > 0.5 else "negative \U0001F614"
                    tk.Label(audio_window, text=f"Transcribed text: {sentiment}").pack()
                except sr.UnknownValueError:
                    tk.Label(audio_window, text="Sorry, couldn't understand the audio.").pack()
                except sr.RequestError as e:
                    tk.Label(audio_window, text=f"Error: {e}").pack()

            self.record_thread = threading.Thread(target=record_audio_thread)
            self.record_thread.start()

        def stop_recording():
            if hasattr(self, 'record_thread') and self.record_thread.is_alive():
                audio_window.title("Stopped Recording")
                recognizer.stop()

        start_button = tk.Button(audio_window, text="Start", command=start_recording)
        start_button.pack()

        stop_button = tk.Button(audio_window, text="Stop", command=stop_recording)
        stop_button.pack()

    def select_image(self):
        image_path = filedialog.askopenfilename(title="Select Image", filetypes=(("Image files", "*.jpg;*.jpeg;*.png"), ("All files", "*.*")))
        if image_path:
            image_window = tk.Toplevel(self)
            image_window.title("Happy or Sad Image Classifier")

        # Load the image
            image = Image.open(image_path)

        # Display the image
            img = ImageTk.PhotoImage(image)
            label = tk.Label(image_window, image=img, bg="#ffffff")
            label.image = img
            label.pack()

        # Make a prediction
            prediction = predict(image)
            print(prediction)
        # Determine the sentiment from the prediction
            sentiment = "Positive 😄" if prediction < 0.5 else "Negative 😔"

        # Display the sentiment
            sentiment_label = tk.Label(image_window, text=f"Prediction: {sentiment}", bg="#ffffff")
            sentiment_label.pack()
if __name__ == "__main__":
    app = InputApplication()
    app.mainloop()
