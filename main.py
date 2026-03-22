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
ENERGY_THRESHOLD = 0.008    
DEVICE = "cuda"             
COMPUTE_TYPE = "float16"

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Real-time Audio Capturer")
        self.root.geometry("900x600")
        
        # UI Setup
        self.text_area = scrolledtext.ScrolledText(
            root, wrap=tk.WORD, font=("Consolas", 11), bg="#1e1e1e", fg="#d4d4d4"
        )
        self.text_area.pack(expand=True, fill='both', padx=10, pady=10)
        
        # Tags for styling
        self.text_area.tag_config("final", foreground="#4ec9b0") 
        self.text_area.tag_config("live", foreground="#ce9178")  
        self.text_area.tag_config("timestamp", foreground="#858585")

        self.audio_queue = Queue()
        
        print("Loading Whisper model...")
        self.model = WhisperModel("medium.en", device=DEVICE, compute_type=COMPUTE_TYPE)
        print("Model loaded.")
        
        self.thread = threading.Thread(target=self.audio_processing_loop, daemon=True)
        self.thread.start()

    def audio_processing_loop(self):
        p = pyaudio.PyAudio()
        
        try:
            # Get the current default WASAPI output device (Speakers or Headphones)
            wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            default_output_index = wasapi_info["defaultOutputDevice"]
            default_speakers = p.get_device_info_by_index(default_output_index)
            
            print(f"Current System Default Output: {default_speakers['name']}")

            # Find the corresponding loopback device for the active output
            loopback_device = None
            for loopback in p.get_loopback_device_info_generator():
                if default_speakers["name"] in loopback["name"]:
                    loopback_device = loopback
                    break

            if not loopback_device:
                print("Matching loopback device not found. Listing all available loopbacks:")
                for lb in p.get_loopback_device_info_generator():
                    print(f" - {lb['name']}")
                return

            print(f"Capturing from: {loopback_device['name']}")

            device_rate = int(loopback_device["defaultSampleRate"])
            channels = loopback_device["maxInputChannels"]
            
            def callback(in_data, frame_count, time_info, status):
                self.audio_queue.put(in_data)
                return (None, pyaudio.paContinue)

            stream = p.open(format=pyaudio.paInt16, 
                            channels=channels, 
                            rate=device_rate,
                            input=True, 
                            input_device_index=loopback_device["index"],
                            stream_callback=callback)

            phrase_time = datetime.utcnow()
            last_sample = np.array([], dtype=np.float32)
            last_speech_time = datetime.utcnow() # Track last time we heard non-silent audio

            while True:
                now = datetime.utcnow()
                if not self.audio_queue.empty():
                    # 1. Gather all available audio from queue
                    current_samples = np.array([], dtype=np.float32)
                    while not self.audio_queue.empty():
                        raw_data = self.audio_queue.get()
                        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                        if channels > 1: 
                            samples = samples.reshape(-1, channels).mean(axis=1)
                        if device_rate != TARGET_RATE: 
                            samples = samples[::int(device_rate / TARGET_RATE)]
                        current_samples = np.concatenate([current_samples, samples])

                    # 2. Check for speech activity (VAD)
                    if np.max(np.abs(current_samples)) > ENERGY_THRESHOLD:
                        last_speech_time = now
                    
                    last_sample = np.concatenate([last_sample, current_samples])

                    # 3. Determine if phrase is complete based on silence duration
                    phrase_complete = False
                    is_silent = (now - last_speech_time > timedelta(seconds=PHRASE_TIMEOUT))
                    if is_silent:
                        phrase_complete = True

                    # 4. Transcribe ONLY if we have actual speech energy in the buffer
                    # This prevents Whisper from hallucinating during silence
                    if len(last_sample) > 0 and np.max(np.abs(last_sample)) > ENERGY_THRESHOLD:
                        # Only transcribe if we haven't already hit silence timeout
                        # Or if we just finished a sentence
                        segments, _ = self.model.transcribe(
                            last_sample, 
                            beam_size=5, 
                            language="en",
                            condition_on_previous_text=False
                        )
                        text = "".join([s.text for s in segments]).strip()

                        # Filter out common hallucinations if the audio is mostly quiet
                        hallucinations = ["thank you", "thanks for watching", "you", "bye", "subscribe", "you.", "thank you.", "bye."]
                        is_hallucination = any(h == text.lower().strip() for h in hallucinations)
                        
                        if text and not (is_silent and is_hallucination):
                            self.update_ui(text, phrase_complete)
                            if phrase_complete:
                                last_sample = np.array([], dtype=np.float32)
                    elif phrase_complete:
                        # If we've been silent long enough, clear the buffer even if no text was found
                        last_sample = np.array([], dtype=np.float32)
                time.sleep(0.1)

        except Exception as e:
            print(f"Audio Loop Error: {e}")
        finally:
            p.terminate()

    def update_ui(self, text, is_final):
        self.root.after(0, self._render_text, text, is_final)
   
    def _render_text(self, text, is_final):
        self.text_area.configure(state='normal')
        
        # 1. Clear existing live text if any (always from the mark to the very end)
        if "live_start" in self.text_area.mark_names():
            self.text_area.delete("live_start", "end")
        
        # Ensure we start on a new line
        current_content = self.text_area.get("1.0", "end-1c")
        if current_content and not current_content.endswith("\n"):
                self.text_area.insert(tk.END, "\n")

        if is_final:
            # 2. Append finalized text with a newline before the timestamp
            ts = datetime.now().strftime("[%H:%M:%S] ")
            
            self.text_area.insert(tk.END, ts, "timestamp")
            self.text_area.insert(tk.END, f"{text}\n", "final")
            # Mark is gone for now as the sentence is finalized
            self.text_area.mark_unset("live_start")
            self.text_area.see(tk.END)
        else:
            # 3. Create/Reset the live section at the end (not using 'insert' which changes on click)
            # We set the mark at 'end-1c' (just before the final newline)
            self.text_area.mark_set("live_start", "end-1c")
            self.text_area.mark_gravity("live_start", tk.LEFT)
            self.text_area.insert(tk.END, f"LIVE >> {text}", "live")
            # Optional: only auto-scroll if user is near the bottom
            self.text_area.see(tk.END)
            
        self.text_area.configure(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()
