import tkinter as tk
from tkinter import filedialog
from pydub import AudioSegment
import numpy as np
from tensorflow.keras.models import model_from_json

class AudioEmotionDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Emotion Detector")
        self.root.geometry("800x600") 

        self.label = tk.Label(root, text="Upload an audio file:")
        self.label.pack(pady=20)  

        self.upload_button = tk.Button(root, text="Upload", command=self.upload_audio)
        self.upload_button.pack(side=tk.BOTTOM, pady=20)  

        self.detect_button = tk.Button(root, text="Detect Emotion", command=self.detect_emotion, state=tk.DISABLED)
        self.detect_button.pack(side=tk.RIGHT, padx=20)  

        self.result_label = tk.Label(root, text="")
        self.result_label.pack(pady=20)  

        self.load_model()

    def upload_audio(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav;*.mp3")])
        if file_path:
            self.audio_path = file_path
            self.detect_button["state"] = tk.NORMAL
            self.label["text"] = f"Audio file: {file_path}"

            self.detect_button.pack_forget()
            self.detect_button.pack(side=tk.RIGHT, padx=20)

    def load_model(self):
        with open('model_a.json', 'r') as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights('Model1.h5')

    def detect_emotion(self):
        try:
            audio = AudioSegment.from_file(self.audio_path)
            audio_array = np.array(audio.get_array_of_samples())

            target_length = 7740
            if len(audio_array) < target_length:
                audio_array = np.pad(audio_array, (0, target_length - len(audio_array)))
            elif len(audio_array) > target_length:
                audio_array = audio_array[:target_length]

            audio_array = audio_array.reshape(1, 1, target_length)
            audio_array = audio_array / np.max(np.abs(audio_array))

            prediction = self.loaded_model.predict(audio_array)
            emotion_labels = ['fear','angry','disgust','neutral','sad','ps','happy']
            emotion_index = np.argmax(prediction)
            detected_emotion = emotion_labels[emotion_index]

            self.result_label["text"] = f"Detected emotion: {detected_emotion}"
        except Exception as e:
            self.result_label["text"] = f"Error: {str(e)}"


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioEmotionDetectorApp(root)
    root.mainloop()
