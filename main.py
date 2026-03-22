import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel
import numpy as np
import threading
from queue import Queue
from datetime import datetime, timedelta
import time
import tkinter as tk
from tkinter import scrolledtext

# --- Configuration ---
TARGET_RATE = 16000
PHRASE_TIMEOUT = 2.0        
ENERGY_THRESHOLD = 0.005    
DEVICE = "cuda"             
COMPUTE_TYPE = "float16"

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Real-time Transcription Engine")
        self.root.geometry("900x600")
        
        # 1. Create a scrolled text area for history
        self.text_area = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, font=("Consolas", 11), bg="#1e1e1e", fg="#d4d4d4"
        )
        self.text_area.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Define text styles (tags)
        self.text_area.tag_config("final", foreground="#4ec9b0") # Confirmed text color
        self.text_area.tag_config("live", foreground="#ce9178")  # Real-time jumping text color
        self.text_area.tag_config("timestamp", foreground="#858585")

        self.audio_queue = Queue()
        
        print("Loading Whisper model, please wait...")
        self.model = WhisperModel("medium.en", device=DEVICE, compute_type=COMPUTE_TYPE)
        print("Model loaded successfully.")
        
        # Start background processing thread
        self.thread = threading.Thread(target=self.audio_processing_loop, daemon=True)
        self.thread.start()

    def audio_processing_loop(self):
        p = pyaudio.PyAudio()
        
        # Find the default WASAPI loopback device
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        
        loopback_device = None
        for loopback in p.get_loopback_device_info_generator():
            if default_speakers["name"] in loopback["name"]:
                loopback_device = loopback
                break

        if not loopback_device:
            print("Error: Loopback device not found.")
            return

        device_rate = int(loopback_device["defaultSampleRate"])
        channels = loopback_device["maxInputChannels"]
        
        def callback(in_data, frame_count, time_info, status):
            self.audio_queue.put(in_data)
            return (None, pyaudio.paContinue)

        stream = p.open(format=pyaudio.paInt16, channels=channels, rate=device_rate,
                        input=True, input_device_index=loopback_device["index"],
                        stream_callback=callback)

        print(">>> System audio capture started. Listening...")

        phrase_time = datetime.utcnow()
        last_sample = np.array([], dtype=np.float32)

        while True:
            now = datetime.utcnow()
            if not self.audio_queue.empty():
                phrase_complete = False
                # If silence duration exceeds timeout, consider phrase finished
                if now - phrase_time > timedelta(seconds=PHRASE_TIMEOUT):
                    phrase_complete = True
                phrase_time = now

                # Combine audio chunks
                while not self.audio_queue.empty():
                    raw_data = self.audio_queue.get()
                    samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                    if channels > 1: 
                        samples = samples.reshape(-1, channels).mean(axis=1)
                    if device_rate != TARGET_RATE: 
                        samples = samples[::int(device_rate / TARGET_RATE)]
                    last_sample = np.concatenate([last_sample, samples])

                # Process only if energy is above threshold
                if np.max(np.abs(last_sample)) > ENERGY_THRESHOLD:
                    segments, _ = self.model.transcribe(last_sample, beam_size=5, language="en")
                    text = "".join([s.text for s in segments]).strip()

                    if text:
                        self.update_ui(text, phrase_complete)
                        if phrase_complete:
                            # Clear buffer for the next phrase
                            last_sample = np.array([], dtype=np.float32)
            time.sleep(0.1)

    def update_ui(self, text, is_final):
        # Schedule the UI update in the main thread
        self.root.after(0, self._render_text, text, is_final)

    def _render_text(self, text, is_final):
        self.text_area.configure(state='normal')
        
        # fix: check if live_start mark exists before deleting
        if "live_start" in self.text_area.mark_names():
            self.text_area.delete("live_start", "end")
        
        if is_final:
            # confirmed text, add timestamp
            ts = datetime.now().strftime("[%H:%M:%S] ")
            self.text_area.insert(tk.END, ts, "timestamp")
            self.text_area.insert(tk.END, f"{text}\n\n", "final")
            # reset live_start mark position after confirmation, avoid interference with next sentence
            self.text_area.mark_unset("live_start")
            self.text_area.see(tk.END) 
        else:
            # real-time changing text: set mark first, then insert text
            self.text_area.mark_set("live_start", "insert")
            self.text_area.mark_gravity("live_start", tk.LEFT) # ensure the mark is fixed at the start
            self.text_area.insert(tk.END, f"LIVE >> {text}", "live")
            self.text_area.see(tk.END) # real-time view the latest typing
            
        self.text_area.configure(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()
