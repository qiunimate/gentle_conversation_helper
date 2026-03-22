from google import genai
import os

class GeminiHelper:
    def __init__(self, api_key=None, model_id="gemini-3-flash-preview"):
        """
        Initializes the Gemini client.
        :param api_key: Your Google AI API Key.
        :param model_id: The Gemini model ID to use.
        """
        # Use provided key or fall back to an environment variable or default
        self.api_key = api_key
        self.model_id = model_id
        self.client = genai.Client(api_key=self.api_key)

    def generate_response(self, prompt_text):
        """
        Generates a text response using the Gemini model.
        :param prompt_text: The user input prompt.
        :return: The generated response text or an error message.
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt_text
            )
            return response.text
        except Exception as e:
            return f"An error occurred in Gemini: {e}"

# Simple usage example
if __name__ == "__main__":
    helper = GeminiHelper()
    result = helper.generate_response("Who is the GOAT badminton player?")
    print(f"Gemini Response: {result}")
