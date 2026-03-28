import tkinter as tk
from tkinter import scrolledtext
from datetime import datetime
import threading
import re
import json
import os

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

        # Dynamic Buttons from JSON
        self.load_dynamic_buttons()

        # Tags for styling
        self.text_area.tag_config("final", foreground="#4ec9b0") 
        self.text_area.tag_config("live", foreground="#ce9178")  
        self.text_area.tag_config("timestamp", foreground="#858585")

        # Initialize Gemini window at startup
        self.ensure_gemini_window()

    def load_dynamic_buttons(self):
        """Loads button configurations from buttons_config.json and creates UI elements."""
        config_path = os.path.join(os.path.dirname(__file__), "buttons_config.json")
        if not os.path.exists(config_path):
            print(f"Warning: {config_path} not found. No dynamic buttons loaded.")
            return

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.button_configs = json.load(f)
            
            for config in self.button_configs:
                btn_name = config.get("name", "Button")
                btn_id = config.get("id")
                shortcut = config.get("shortcut")
                
                # Create button
                btn = tk.Button(
                    self.control_frame, text=btn_name, 
                    command=lambda c=config: self.handle_gemini_request(c),
                    bg="#333333", fg="white", activebackground="#444444", activeforeground="white",
                    relief=tk.FLAT, padx=10, pady=5
                )
                btn.pack(side=tk.LEFT, padx=(0, 5))

                # Bind shortcut if provided
                if shortcut:
                    self.root.bind(shortcut, lambda e, c=config: self.handle_gemini_request(c))
                    
        except Exception as e:
            print(f"Error loading button configuration: {e}")

    def get_last_200_words(self):
        """Helper to extract and clean the last 200 words from the text area."""
        full_raw_text = self.text_area.get("1.0", "end-1c")
        
        # Clean timestamps [HH:MM:SS] and "LIVE >>" markers
        cleaned_text = re.sub(r"\[\d{2}:\d{2}:\d{2}\]\s*", "", full_raw_text)
        cleaned_text = cleaned_text.replace("LIVE >>", "")
        
        # Split into words and get last 200
        words = cleaned_text.split()
        context_words = words[-200:] if len(words) > 200 else words
        return " ".join(context_words), len(context_words), context_words

    def handle_gemini_request(self, config):
        """Unified handler for Gemini requests based on JSON configuration."""
        if not self.gemini_helper:
            print("Gemini Helper not initialized!")
            return

        # 1. Ensure Gemini window is open
        self.ensure_gemini_window()

        # 2. Get context
        context_string, word_count, context_words = self.get_last_200_words()

        if not context_string:
            self.gemini_text_area.configure(state='normal')
            ts = datetime.now().strftime("[%H:%M:%S] ")
            self.gemini_text_area.insert(tk.END, ts, "timestamp")
            self.gemini_text_area.insert(tk.END, "No text found for context.\n", "gemini_label")
            self.gemini_text_area.configure(state='disabled')
            self.gemini_text_area.see(tk.END)
            return

        # 3. Show UI feedback in Gemini window
        start_snippet = " ".join(context_words[:10]) if len(context_words) > 10 else " ".join(context_words)
        end_snippet = " ".join(context_words[-15:]) if len(context_words) > 15 else ""
        display_summary = f"\"{start_snippet} ... {end_snippet}\"" if end_snippet else f"\"{start_snippet}\""

        self.gemini_text_area.configure(state='normal')
        ts = datetime.now().strftime("[%H:%M:%S] ")
        self.gemini_text_area.insert(tk.END, ts, "timestamp")
        label = f"[{config.get('name')} - Last {word_count} words]: {display_summary}\n"
        self.gemini_text_area.insert(tk.END, label, "gemini_label")
        self.gemini_text_area.configure(state='disabled')
        self.gemini_text_area.update_idletasks()
        self.gemini_text_area.see(tk.END)

        # 4. Call Gemini in a thread
        def call_api():
            try:
                prompt_template = config.get("prompt", "{context}")
                final_prompt = prompt_template.replace("{context}", context_string)
                
                response = self.gemini_helper.generate_response(final_prompt)
                self.root.after(0, self._display_gemini_response, response)
            except Exception as e:
                self.gemini_text_area.insert(tk.END, f"Error: {str(e)}\n", "gemini_label")
                self.gemini_text_area.see(tk.END)

        threading.Thread(target=call_api, daemon=True).start()

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
