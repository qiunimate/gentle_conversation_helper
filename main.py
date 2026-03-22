import pyaudiowpatch as pyaudio
from faster_whisper import WhisperModel
import numpy as np
from queue import Queue
from datetime import datetime, timedelta
import time
import sys
from colorama import init, Fore

init(autoreset=True)

# --- 参数配置 ---
TARGET_RATE = 16000
CHUNK_SIZE = 1024           # the size of each audio chunk
PHRASE_TIMEOUT = 2.0        # the timeout for a phrase (2 seconds of silence)
RECORD_TIMEOUT = 1.0        # the interval for processing the buffer (1 second)
ENERGY_THRESHOLD = 0.005    # the energy threshold for detecting sound (0.005)
DEVICE = "cuda"             # use GPU (RTX 3060)
COMPUTE_TYPE = "float16"

# Load the model
print(f"{Fore.YELLOW}loading Whisper...{Fore.RESET}")
model = WhisperModel("medium.en", device=DEVICE, compute_type=COMPUTE_TYPE)

def start_engine():
    p = pyaudio.PyAudio()
    
    # 1. find the default speakers (WASAPI Loopback)
    wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
    default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
    
    loopback_device = None
    for loopback in p.get_loopback_device_info_generator():
        if default_speakers["name"] in loopback["name"]:
            loopback_device = loopback
            break

    if not loopback_device:
        print(f"{Fore.RED}could not find loopback device!{Fore.RESET}")
        return

    # 2. audio configuration
    device_rate = int(loopback_device["defaultSampleRate"])
    channels = loopback_device["maxInputChannels"]
    
    data_queue = Queue()
    
    def callback(in_data, frame_count, time_info, status):
        data_queue.put(in_data)
        return (None, pyaudio.paContinue)

    # 3. start the audio stream
    stream = p.open(format=pyaudio.paInt16, 
                    channels=channels, 
                    rate=device_rate,
                    input=True, 
                    input_device_index=loopback_device["index"],
                    stream_callback=callback)

    print(f"{Fore.CYAN}>>> system audio capture started. listening...{Fore.RESET}\n")

    # --- logic variables ---
    phrase_time = datetime.utcnow()
    last_sample = np.array([], dtype=np.float32)
    
    try:
        while True:
            now = datetime.utcnow()
            
            if not data_queue.empty():
                phrase_complete = False
                
                # check if the phrase is complete
                if now - phrase_time > timedelta(seconds=PHRASE_TIMEOUT):
                    phrase_complete = True
                
                phrase_time = now

                # extract all audio data from the queue
                while not data_queue.empty():
                    raw_data = data_queue.get()
                    # convert to floating point and normalize
                    samples = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                    # convert multi-channel to mono
                    if channels > 1:
                        samples = samples.reshape(-1, channels).mean(axis=1)
                    # resample to 16000Hz
                    if device_rate != TARGET_RATE:
                        samples = samples[::int(device_rate / TARGET_RATE)]
                    
                    last_sample = np.concatenate([last_sample, samples])

                # only transcribe when the volume exceeds the threshold to avoid phantom listening
                if np.max(np.abs(last_sample)) > ENERGY_THRESHOLD:
                    # transcribe the audio
                    segments, _ = model.transcribe(last_sample, beam_size=5, language="en")
                    text = "".join([s.text for s in segments]).strip()

                    # real-time output (refresh on the same line)
                    sys.stdout.write(f"\r{Fore.GREEN}Listening: {text}\033[K")
                    sys.stdout.flush()

                    # if the sentence is complete, print the final result and clear the buffer
                    if phrase_complete and text:
                        print(f"\n{Fore.WHITE}✓ {text}{Fore.RESET}")
                        last_sample = np.array([], dtype=np.float32)
                
            else:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}stopping...{Fore.RESET}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    start_engine()
