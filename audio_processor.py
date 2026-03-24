import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel
import numpy as np
import threading
from queue import Queue
from datetime import datetime, timedelta
import time

class AudioProcessor:
    def __init__(self, model_name="medium.en", device="cuda", compute_type="float16", 
                 target_rate=16000, phrase_timeout=1.0, energy_threshold=0.008):
        self.target_rate = target_rate
        self.phrase_timeout = phrase_timeout
        self.energy_threshold = energy_threshold
        self.audio_queue = Queue()
        self.is_running = False
        
        print(f"Loading Whisper model ({model_name})...")
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        print("Model loaded.")

    def find_loopback_device(self, p):
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_output_index = wasapi_info["defaultOutputDevice"]
        default_speakers = p.get_device_info_by_index(default_output_index)
        
        print(f"--- WASAPI Audio Devices ---")
        print(f"Default Output Device: {default_speakers['name']} (Index: {default_output_index})")
        
        loopback_devices = list(p.get_loopback_device_info_generator())
        if not loopback_devices:
            print("Error: No WASAPI loopback devices found. Please ensure WASAPI is supported on your system.")
            return None

        print(f"Found {len(loopback_devices)} loopback devices:")
        target_device = None
        
        # 1. First try to find a match for the system's default output device
        speaker_name = default_speakers["name"].split('(')[0].strip()
        for loopback in loopback_devices:
            print(f" - {loopback['name']} (Index: {loopback['index']})")
            if speaker_name in loopback["name"]:
                target_device = loopback
                print(f"Match found for default output! Using: {loopback['name']}")
                break
        
        # 2. If no match for default, prioritize any device with "Headphones" in name
        if not target_device:
            for loopback in loopback_devices:
                if "Headphones" in loopback["name"] or "Headset" in loopback["name"]:
                    target_device = loopback
                    print(f"Match found for Headphones/Headset! Using: {loopback['name']}")
                    break
                    
        # 3. Final Fallback: Just take the first loopback device
        if not target_device:
            target_device = loopback_devices[0]
            print(f"Warning: No exact match found. Falling back to first loopback: {target_device['name']}")
            
        return target_device

    def start_capture(self, callback_ui):
        self.is_running = True
        self.thread = threading.Thread(target=self._audio_loop, args=(callback_ui,), daemon=True)
        self.thread.start()

    def _audio_loop(self, callback_ui):
        p = pyaudio.PyAudio()
        try:
            loopback_device = self.find_loopback_device(p)
            if not loopback_device:
                print("Loopback device not found.")
                return

            device_rate = int(loopback_device["defaultSampleRate"])
            channels = loopback_device["maxInputChannels"]
            
            def stream_callback(in_data, frame_count, time_info, status):
                self.audio_queue.put(in_data)
                return (None, pyaudio.paContinue)

            stream = p.open(format=pyaudio.paInt16, 
                            channels=channels, 
                            rate=device_rate,
                            input=True, 
                            input_device_index=loopback_device["index"],
                            stream_callback=stream_callback)

            last_sample = np.array([], dtype=np.float32)
            last_speech_time = datetime.utcnow()

            while self.is_running:
                now = datetime.utcnow()
                if not self.audio_queue.empty():
                    current_samples = np.array([], dtype=np.float32)
                    while not self.audio_queue.empty():
                        raw_data = self.audio_queue.get()
                        samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                        if channels > 1: samples = samples.reshape(-1, channels).mean(axis=1)
                        
                        # Robust resampling: using linear interpolation if rates don't match
                        if device_rate != self.target_rate and len(samples) > 0:
                            new_len = int(len(samples) * self.target_rate / device_rate)
                            if new_len > 0:
                                x_old = np.linspace(0, 1, len(samples))
                                x_new = np.linspace(0, 1, new_len)
                                samples = np.interp(x_new, x_old, samples)
                            else:
                                samples = np.array([], dtype=np.float32)
                            
                        current_samples = np.concatenate([current_samples, samples])

                    if np.max(np.abs(current_samples)) > self.energy_threshold:
                        last_speech_time = now
                    
                    last_sample = np.concatenate([last_sample, current_samples])
                    
                    is_silent = (now - last_speech_time > timedelta(seconds=self.phrase_timeout))
                    phrase_complete = is_silent

                    if len(last_sample) > 0 and np.max(np.abs(last_sample)) > self.energy_threshold:
                        segments, _ = self.model.transcribe(
                            last_sample, beam_size=5, language="en", condition_on_previous_text=False
                        )
                        text = "".join([s.text for s in segments]).strip()

                        hallucinations = ["thank you", "thanks for watching", "you", "bye", "subscribe", "you.", "thank you.", "bye."]
                        is_hallucination = any(h == text.lower().strip() for h in hallucinations)
                        
                        if text and not (is_silent and is_hallucination):
                            callback_ui(text, phrase_complete)
                            if phrase_complete:
                                last_sample = np.array([], dtype=np.float32)
                    elif phrase_complete:
                        last_sample = np.array([], dtype=np.float32)
                
                time.sleep(0.1)
        finally:
            self.is_running = False
            p.terminate()
