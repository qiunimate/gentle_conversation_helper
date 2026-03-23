import tkinter as tk
from tkinter import scrolledtext
from datetime import datetime

class UIManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Whisper Real-time Audio Capturer")
        self.root.geometry("900x700")
        self.gemini_helper = None # To be set by main.py
        
        # Gemini Window references
        self.gemini_window = None
        self.gemini_text_area = None

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

        # Initialize Gemini window at startup
        self.ensure_gemini_window()

    def ensure_gemini_window(self):
        """Creates or restores the separate Gemini window."""
        if self.gemini_window is None or not self.gemini_window.winfo_exists():
            self.gemini_window = tk.Toplevel(self.root)
            self.gemini_window.title("Gemini Assistant - Q&A")
            self.gemini_window.geometry("600x600")
            self.gemini_window.configure(bg="#1e1e1e")
            
            # Gemini window text area setup
            self.gemini_text_area = scrolledtext.ScrolledText(
                self.gemini_window, wrap=tk.WORD, font=("Consolas", 11), 
                bg="#1e1e1e", fg="#d4d4d4", insertbackground="white"
            )
            self.gemini_text_area.pack(expand=True, fill='both', padx=10, pady=10)
            
            # Tags for styling Gemini window
            self.gemini_text_area.tag_config("gemini_label", foreground="#c586c0", font=("Consolas", 11, "bold"))
            self.gemini_text_area.tag_config("gemini_text", foreground="#9cdcfe", background="#2a2a2a")
            self.gemini_text_area.tag_config("timestamp", foreground="#858585")
            self.gemini_text_area.configure(state='disabled')
        
        # Bring to front
        self.gemini_window.deiconify()
        self.gemini_window.lift()

    def on_ask_gemini(self):
        if not self.gemini_helper:
            print("Gemini Helper not initialized!")
            return

        # 1. Ensure Gemini window is open
        self.ensure_gemini_window()

        # 2. Extract context from text area (including live text)
        full_text = self.text_area.get("1.0", "end-1c")
        lines = [l.strip() for l in full_text.split("\n") if l.strip()]
        
        if not lines:
            print("No text found to send to Gemini.")
            return

        # Get the very last line, whether it's LIVE or Finalized
        last_line = lines[-1]
        
        import re
        # If it's a LIVE line, strip the "LIVE >> " prefix
        if last_line.startswith("LIVE >>"):
            prompt = last_line.replace("LIVE >>", "").strip()
        else:
            # If it's a finalized line, strip the timestamp [HH:MM:SS]
            prompt = re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s*", "", last_line)

        # If the last line was empty after stripping (e.g. just "LIVE >> "), 
        # try the previous finalized line if available
        if not prompt and len(lines) > 1:
            prev_line = lines[-2]
            prompt = re.sub(r"^\[\d{2}:\d{2}:\d{2}\]\s*", "", prev_line)

        if not prompt:
            print("No valid prompt extracted.")
            return

        # 3. Show question in Gemini window
        self.gemini_text_area.configure(state='normal')
        ts = datetime.now().strftime("[%H:%M:%S] ")
        self.gemini_text_area.insert(tk.END, ts, "timestamp")
        self.gemini_text_area.insert(tk.END, f"[Question]: \"{prompt}\"...\n", "gemini_label")
        self.gemini_text_area.configure(state='disabled')
        self.gemini_text_area.update_idletasks() # Force UI update
        self.gemini_text_area.see(tk.END)

        # 4. Call Gemini in a separate thread
        import threading
        def call_api():
            try:
                print(f"Calling Gemini with prompt: {prompt}")
                response = self.gemini_helper.generate_response(f"The following is a part of an interview transcription. Please answer the last question concisely as if you are the interviewee. Context: {prompt}")
                print(f"Gemini response received: {response[:50]}...")
                self.root.after(0, self._display_gemini_response, response)
            except Exception as e:
                print(f"Error calling Gemini API: {e}")
                self.root.after(0, lambda: self._display_gemini_response(f"Error: {str(e)}"))

        threading.Thread(target=call_api, daemon=True).start()

    def _display_gemini_response(self, response):
        self.ensure_gemini_window()
        self.gemini_text_area.configure(state='normal')
        self.gemini_text_area.insert(tk.END, f"[Answer]: ", "gemini_label")
        self.gemini_text_area.insert(tk.END, f"{response}\n\n", "gemini_text")
        self.gemini_text_area.configure(state='disabled')
        self.gemini_text_area.see(tk.END)

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
