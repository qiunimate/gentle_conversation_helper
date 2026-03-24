from google import genai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class GeminiHelper:
    def __init__(self, api_key=None, model_id="gemini-3-flash-preview", system_instruction=None):
        """
        Initializes the Gemini client.
        :param api_key: Your Google AI API Key.
        :param model_id: The Gemini model ID to use.
        :param system_instruction: The system prompt (persona, CV, etc.)
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        
        if not self.api_key:
            raise ValueError("No Gemini API key found. Please provide one or set GEMINI_API_KEY in your .env file.")
            
        self.model_id = model_id
        self.system_instruction = system_instruction
        self.client = genai.Client(api_key=self.api_key)

    def generate_response(self, prompt_text):
        """
        Generates a text response using the Gemini model.
        :param prompt_text: The user input prompt.
        :return: The generated response text or an error message.
        """
        try:
            # Using the new SDK's config for system_instruction
            config = {}
            if self.system_instruction:
                config['system_instruction'] = self.system_instruction
                
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt_text,
                config=config
            )
            return response.text
        except Exception as e:
            return f"An error occurred in Gemini: {e}"

# Simple usage example
if __name__ == "__main__":
    helper = GeminiHelper()
    result = helper.generate_response("What is the capital of France?")
    print(f"Gemini Response:")
    print(result)
