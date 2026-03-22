import tkinter as tk
from tkinter import scrolledtext
from datetime import datetime

class UIManager:
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

    def update_ui(self, text, is_final):
        self.root.after(0, self._render_text, text, is_final)
   
    def _render_text(self, text, is_final):
        self.text_area.configure(state='normal')
        
        # 1. Clear existing live text if any
        if "live_start" in self.text_area.mark_names():
            self.text_area.delete("live_start", "end")
        
        # Ensure we start on a new line
        current_content = self.text_area.get("1.0", "end-1c")
        if current_content and not current_content.endswith("\n"):
            self.text_area.insert(tk.END, "\n")

        if is_final:
            # 2. Append finalized text
            ts = datetime.now().strftime("[%H:%M:%S] ")
            self.text_area.insert(tk.END, ts, "timestamp")
            self.text_area.insert(tk.END, f"{text}\n", "final")
            self.text_area.mark_unset("live_start")
            self.text_area.see(tk.END)
        else:
            # 3. Update live text at the end
            self.text_area.mark_set("live_start", "end-1c")
            self.text_area.mark_gravity("live_start", tk.LEFT)
            self.text_area.insert(tk.END, f"LIVE >> {text}", "live")
            self.text_area.see(tk.END)
            
        self.text_area.configure(state='disabled')
