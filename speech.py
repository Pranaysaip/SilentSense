import tkinter as tk
import threading
import speech_recognition as sr

class SpeechToTextApp:
    def __init__(self, master):
        self.master = master
        master.title("Speech to Text")

        # Full screen
        master.attributes("-fullscreen", True)

        # Background color
        master.config(bg="#f0f0f0")

        # Title
        self.title_label = tk.Label(master, text="Speech to Text Converter", font=("Segoe UI", 36, "bold"), bg="#f0f0f0", fg="#333")
        self.title_label.pack(pady=50)

        # Transcription area
        self.transcription_text = tk.Text(master, height=10, width=50, font=("Segoe UI", 16), bg="#fff", relief="flat", borderwidth=2)
        self.transcription_text.pack(pady=20)

        # Start and Stop buttons
        self.button_frame = tk.Frame(master, bg="#f0f0f0")
        self.button_frame.pack()

        self.start_button = tk.Button(self.button_frame, text="Start", command=self.start_transcription, font=("Segoe UI", 20), bg="#4caf50", fg="#fff", relief="flat", borderwidth=0)
        self.start_button.pack(side=tk.LEFT, padx=20, pady=10)

        self.stop_button = tk.Button(self.button_frame, text="Stop", command=self.stop_transcription, font=("Segoe UI", 20), bg="#f44336", fg="#fff", relief="flat", borderwidth=0, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=20, pady=10)

        # Variable to control the transcription loop
        self.transcribing = False

        # Lock to synchronize access to the text widget
        self.lock = threading.Lock()

    def start_transcription(self):
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.transcription_text.delete("1.0", tk.END)  # Clear previous transcription
        self.transcribing = True
        threading.Thread(target=self.transcribe_loop).start()

    def stop_transcription(self):
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.transcribing = False

    def transcribe_loop(self):
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()

        while self.transcribing:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)

            try:
                transcription = recognizer.recognize_google(audio)
                self.lock.acquire()
                self.transcription_text.insert(tk.END, transcription + "\n")
                self.transcription_text.see(tk.END)  # Scroll to the end
            except sr.UnknownValueError:
                pass  # Ignore if audio is not understood
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))
            finally:
                self.lock.release()

def main():
    root = tk.Tk()
    app = SpeechToTextApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
