#!/usr/bin/env python3
"""
Simple test version of DBP Bot without complex LangChain dependencies
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    print("[WARNING] langchain-groq not available. Using fallback responses.")
    GROQ_AVAILABLE = False

class SimpleChatbot:
    def __init__(self):
        # Load environment
        load_dotenv()
        
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = None
        
        if GROQ_AVAILABLE and self.groq_api_key:
            try:
                self.llm = ChatGroq(
                    model_name="gemma2-9b-it",
                    groq_api_key=self.groq_api_key,
                    temperature=0.7,
                    max_tokens=500
                )
                print("[INFO] Successfully connected to Groq!")
            except Exception as e:
                print(f"[ERROR] Failed to initialize Groq: {e}")
        else:
            print("[INFO] Using fallback responses (no Groq connection)")
    
    def get_response(self, message: str) -> str:
        """Get a response to user input"""
        
        # Simple state detection
        stressed_keywords = ["stress", "anxious", "worried", "upset", "overwhelmed", "panic"]
        is_stressed = any(word in message.lower() for word in stressed_keywords)
        
        if self.llm:
            try:
                if is_stressed:
                    prompt = f"You are a compassionate mental health support bot. The user seems stressed and said: '{message}'. Respond with empathy and gentle guidance. Keep it under 100 words."
                else:
                    prompt = f"You are a friendly wellness companion. The user said: '{message}'. Respond in a warm, supportive way. Keep it under 100 words."
                
                response = self.llm.invoke(prompt)
                return response.content if hasattr(response, 'content') else str(response)
                
            except Exception as e:
                print(f"[ERROR] LLM call failed: {e}")
                return self._fallback_response(message, is_stressed)
        else:
            return self._fallback_response(message, is_stressed)
    
    def _fallback_response(self, message: str, is_stressed: bool) -> str:
        """Fallback responses when LLM is not available"""
        if is_stressed:
            return f"I hear that you're going through something difficult. It's completely normal to feel this way, and I want you to know that you're not alone. Take a deep breath - you're stronger than you think."
        else:
            return f"Thank you for sharing that with me. I'm here to listen and support you. How are you feeling about everything right now?"

def main():
    print("🤖 Simple DBP Bot Test")
    print("Type 'quit' or 'exit' to end the conversation")
    print("-" * 50)
    
    bot = SimpleChatbot()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nBot: Take care! Remember, you're doing great. 💙")
                break
                
            if not user_input:
                continue
                
            response = bot.get_response(user_input)
            print(f"\nBot: {response}")
            
        except KeyboardInterrupt:
            print("\n\nBot: Take care! 💙")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Bot: I'm having some technical difficulties, but I'm still here for you.")

if __name__ == "__main__":
    main()