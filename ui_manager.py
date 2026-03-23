import tkinter as tk
from tkinter import scrolledtext
from datetime import datetime

class UIManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Real-time Audio Capturer")
        self.root.geometry("900x700")
        self.gemini_helper = None # To be set by main.py
        
        # Main Layout
        self.main_frame = tk.Frame(root, bg="#1e1e1e")
        self.main_frame.pack(expand=True, fill='both')

        # Text Area Setup
        self.text_area = scrolledtext.ScrolledText(
            self.main_frame, wrap=tk.WORD, font=("Consolas", 11), bg="#1e1e1e", fg="#d4d4d4",
            insertbackground="white"
        )
        self.text_area.pack(expand=True, fill='both', padx=10, pady=(10, 5))
        
        # Bottom Control Frame
        self.control_frame = tk.Frame(self.main_frame, bg="#1e1e1e")
        self.control_frame.pack(fill='x', padx=10, pady=(0, 10))

        self.ask_button = tk.Button(
            self.control_frame, text="Ask Gemini (Alt+G)", command=self.on_ask_gemini,
            bg="#333333", fg="white", activebackground="#444444", activeforeground="white",
            relief=tk.FLAT, padx=10, pady=5
        )
        self.ask_button.pack(side=tk.LEFT)

        # Bind shortcut
        self.root.bind('<Alt-g>', lambda e: self.on_ask_gemini())

        # Tags for styling
        self.text_area.tag_config("final", foreground="#4ec9b0") 
        self.text_area.tag_config("live", foreground="#ce9178")  
        self.text_area.tag_config("timestamp", foreground="#858585")
        self.text_area.tag_config("gemini_label", foreground="#c586c0", font=("Consolas", 11, "bold"))
        self.text_area.tag_config("gemini_text", foreground="#9cdcfe", background="#2a2a2a")

    def on_ask_gemini(self):
        if not self.gemini_helper:
            return

        # 1. Extract context from text area
        # We take the last few lines to find the "question"
        full_text = self.text_area.get("1.0", "end-1c")
        lines = [l.strip() for l in full_text.split("\n") if l.strip() and not l.startswith("LIVE >>")]
        
        if not lines:
            return

        # Simple logic: take the last non-empty finalized line as the question
        last_context = lines[-1]
        
        # If the line starts with a timestamp [HH:MM:SS], strip it
        import re
        prompt = re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s*", "", last_context)

        # 2. Show loading state in UI
        self.text_area.configure(state='normal')
        self.text_area.insert(tk.END, f"\n[Question]: \"{prompt}\"...\n", "gemini_label")
        self.text_area.configure(state='disabled')
        self.text_area.see(tk.END)

        # 3. Call Gemini in a separate thread to avoid freezing UI
        import threading
        def call_api():
            response = self.gemini_helper.generate_response(f"The following is a part of an interview transcription. Please answer the last question concisely as if you are the interviewee. Context: {prompt}")
            self.root.after(0, self._display_gemini_response, response)

        threading.Thread(target=call_api, daemon=True).start()

    def _display_gemini_response(self, response):
        self.text_area.configure(state='normal')
        # Replace the thinking message or just append
        self.text_area.insert(tk.END, f"[Answer]: ", "gemini_label")
        self.text_area.insert(tk.END, f"{response}\n\n", "gemini_text")
        self.text_area.configure(state='disabled')
        self.text_area.see(tk.END)

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
