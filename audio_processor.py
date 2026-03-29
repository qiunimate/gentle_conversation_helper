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
        self.output_queue = Queue()
        self.input_queue = Queue()
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
            # 1. Setup Output Loopback Device
            loopback_device = self.find_loopback_device(p)
            if not loopback_device:
                print("Loopback device not found.")
                return

            out_rate = int(loopback_device["defaultSampleRate"])
            out_channels = loopback_device["maxInputChannels"]
            
            def out_callback(in_data, frame_count, time_info, status):
                self.output_queue.put(in_data)
                return (None, pyaudio.paContinue)

            out_stream = p.open(format=pyaudio.paInt16, 
                                channels=out_channels, 
                                rate=out_rate,
                                input=True, 
                                input_device_index=loopback_device["index"],
                                stream_callback=out_callback)

            # 2. Setup Input (Microphone) Device
            try:
                input_device = p.get_default_input_device_info()
                in_rate = int(input_device["defaultSampleRate"])
                in_channels = input_device["maxInputChannels"]
                print(f"Default Input Device: {input_device['name']} (Index: {input_device['index']})")
                
                def in_callback(in_data, frame_count, time_info, status):
                    self.input_queue.put(in_data)
                    return (None, pyaudio.paContinue)

                in_stream = p.open(format=pyaudio.paInt16,
                                   channels=in_channels,
                                   rate=in_rate,
                                   input=True,
                                   input_device_index=input_device["index"],
                                   stream_callback=in_callback)
            except Exception as e:
                print(f"Warning: Could not open default input device: {e}")
                in_stream = None

            last_sample = np.array([], dtype=np.float32)
            last_speech_time = datetime.utcnow()
            current_source = "system" # Default source

            def process_raw_data(raw_data, rate, channels):
                samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                if channels > 1: samples = samples.reshape(-1, channels).mean(axis=1)
                if rate != self.target_rate and len(samples) > 0:
                    new_len = int(len(samples) * self.target_rate / rate)
                    if new_len > 0:
                        x_old = np.linspace(0, 1, len(samples))
                        x_new = np.linspace(0, 1, new_len)
                        samples = np.interp(x_new, x_old, samples)
                    else:
                        samples = np.array([], dtype=np.float32)
                return samples

            while self.is_running:
                now = datetime.utcnow()
                
                # Drain both queues
                out_samples = np.array([], dtype=np.float32)
                while not self.output_queue.empty():
                    out_samples = np.concatenate([out_samples, process_raw_data(self.output_queue.get(), out_rate, out_channels)])
                
                in_samples = np.array([], dtype=np.float32)
                while not self.input_queue.empty():
                    in_samples = np.concatenate([in_samples, process_raw_data(self.input_queue.get(), in_rate, in_channels)])

                # Mix them
                max_len = max(len(out_samples), len(in_samples))
                if max_len > 0:
                    mixed = np.zeros(max_len, dtype=np.float32)
                    out_energy = np.max(np.abs(out_samples)) if len(out_samples) > 0 else 0
                    in_energy = np.max(np.abs(in_samples)) if len(in_samples) > 0 else 0
                    
                    if out_energy > in_energy and out_energy > self.energy_threshold:
                        current_source = "system"
                    elif in_energy > out_energy and in_energy > self.energy_threshold:
                        current_source = "mic"

                    if len(out_samples) > 0: mixed[:len(out_samples)] += out_samples
                    if len(in_samples) > 0: mixed[:len(in_samples)] += in_samples
                    
                    current_samples = mixed

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
                            callback_ui(text, phrase_complete, current_source)
                            if phrase_complete:
                                last_sample = np.array([], dtype=np.float32)
                                current_source = "system" # Reset source
                    elif phrase_complete:
                        last_sample = np.array([], dtype=np.float32)
                        current_source = "system" # Reset source
                
                time.sleep(0.1)
        finally:
            self.is_running = False
            p.terminate()
