import tkinter as tk
from audio_processor import AudioProcessor
from ui_manager import UIManager
from gemini_helper import GeminiHelper

# --- Configuration ---
TARGET_RATE = 16000         # Target sample rate for Whisper model
PHRASE_TIMEOUT = 1.0        # Timeout for complete phrase in seconds
ENERGY_THRESHOLD = 0.01    # Energy threshold for voice activity detection
MODEL_NAME = "base.en"    # Name of the Whisper model to use
DEVICE = "cuda"             
COMPUTE_TYPE = "float16"
INSTRUCTION_FILE = "system_instruction.txt"

def load_instruction(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error loading instruction file: {e}, using blank instruction.")
        return ""

if __name__ == "__main__":
    root = tk.Tk()
    
    # Initialize UI and Audio components
    ui = UIManager(root)
    
    # Load system instruction from external file
    system_instruction = load_instruction(INSTRUCTION_FILE)
    
    # Initialize Gemini with system instruction
    try:
        ui.gemini_helper = GeminiHelper(system_instruction=system_instruction)
    except Exception as e:
        print(f"Warning: Gemini not initialized: {e}")

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
