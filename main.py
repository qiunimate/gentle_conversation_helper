import tkinter as tk
from audio_processor import AudioProcessor
from ui_manager import UIManager

# --- Configuration ---
TARGET_RATE = 16000         # Target sample rate for Whisper model
PHRASE_TIMEOUT = 1.0        # Timeout for complete phrase in seconds
ENERGY_THRESHOLD = 0.008    # Energy threshold for voice activity detection
MODEL_NAME = "base.en"    # Name of the Whisper model to use
DEVICE = "cuda"             
COMPUTE_TYPE = "float16"

if __name__ == "__main__":
    root = tk.Tk()
    
    # Initialize UI and Audio components
    ui = UIManager(root)
    processor = AudioProcessor(
        model_name=MODEL_NAME, 
        device=DEVICE, 
        compute_type=COMPUTE_TYPE,
        target_rate=TARGET_RATE,
        phrase_timeout=PHRASE_TIMEOUT,
        energy_threshold=ENERGY_THRESHOLD
    )
    
    # Link audio output to UI updates
    processor.start_capture(ui.update_ui)
    
    # Start the Tkinter loop
    root.mainloop()
