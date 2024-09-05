import tkinter as tk
from tkinter import ttk, Text, Toplevel
from PIL import Image, ImageTk
import speech_recognition as sr
from threading import Thread
from gtts import gTTS
import os

class TextToSpeechApp:
    def __init__(self, master):
        self.master = master
        master.title("Text-To-Speech and Speech-To-Text Converter")
        master.attributes("-fullscreen", True)
        master.configure(bg="white")

        # Load images
        self.stot_img = ImageTk.PhotoImage(Image.open("TtoS.png").resize((300, 168)))
        self.ttost_img = ImageTk.PhotoImage(Image.open("StoT.png").resize((300, 168)))

        # Title label
        self.label = tk.Label(master, text="Text-To-Speech and Speech-To-Text Converter",
                              font=("Times New Roman", 24), bg="white")
        self.label.pack(pady=(50, 20))

        # Text-to-Speech button with image and text
        self.text_to_speech_button = ttk.Button(master, text="Text-To-Speech", style='TButton',
                                                command=self.text_to_speech, image=self.stot_img, compound="top")
        self.text_to_speech_button.image = self.stot_img
        self.text_to_speech_button.place(relx=0.25, rely=0.5, anchor=tk.CENTER)

        # Speech-to-Text button with image and text
        self.speech_to_text_button = ttk.Button(master, text="Speech-To-Text", style='TButton',
                                                command=self.speech_to_text, image=self.ttost_img, compound="top")
        self.speech_to_text_button.image = self.ttost_img
        self.speech_to_text_button.place(relx=0.75, rely=0.5, anchor=tk.CENTER)

        # Exit button
        self.exit_button = tk.Button(master, text="Exit", font=("Arial", 12), bg="red", fg="white", command=master.quit)
        self.exit_button.pack(side=tk.BOTTOM, pady=20, fill=tk.X)

    def text_to_speech(self):
        text_to_speech_window = Toplevel(self.master)
        text_to_speech_window.title("Text-to-Speech Converter")
        text_to_speech_window.geometry("600x400")
        text_to_speech_window.configure(bg="white")

        tk.Label(text_to_speech_window, text="Text-to-Speech Converter", font=("Times New Roman", 24),
                 bg="white").pack(pady=20)

        self.text_input = Text(text_to_speech_window, height=5, width=50, font=12)
        self.text_input.pack(pady=20)

        speak_button = ttk.Button(text_to_speech_window, text="Listen", command=self.say_text, style='TButton')
        speak_button.pack(pady=20)

    def speech_to_text(self):
        speech_to_text_window = Toplevel(self.master)
        speech_to_text_window.title("Speech-to-Text Converter")
        speech_to_text_window.geometry("600x400")
        speech_to_text_window.configure(bg="white")

        tk.Label(speech_to_text_window, text="Speech-to-Text Converter", font=("Times New Roman", 24),
                 bg="white").pack(pady=20)

        self.speech_output = Text(speech_to_text_window, font=12, height=10, width=50)
        self.speech_output.pack(pady=20)

        record_button = ttk.Button(speech_to_text_window, text="Record", command=self.record_voice, style='TButton')
        record_button.pack(pady=20)

    def say_text(self):
        language = "en"
        text = self.text_input.get(1.0, tk.END)
        speech = gTTS(text=text, lang=language, slow=False)
        speech.save("text.mp3")
        os.system("start text.mp3")

    def record_voice(self):
        def record():
            r = sr.Recognizer()
            with sr.Microphone() as source:
                audio = r.listen(source)
                try:
                    text = r.recognize_google(audio, language="en-IN")
                    self.speech_output.insert(tk.END, text)
                except sr.UnknownValueError:
                    pass  # Ignore if audio is not understood
                except sr.RequestError as e:
                    print("Could not request results from Google Speech Recognition service; {0}".format(e))

        Thread(target=record).start()


def main():
    root = tk.Tk()
    style = ttk.Style(root)
    style.configure('TButton', background='black', foreground='black', font=('Helvetica', 14, 'bold'))
    app = TextToSpeechApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
