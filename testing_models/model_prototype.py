import pdfplumber
import spacy # Import spacy
from spacy import displacy
import os
import json
import random
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

# Load environment variables (for API key, if not using Canvas's auto-injection)
load_dotenv()

# Define TEXT_FILE_FOR_KNOWLEDGE as it's used in run_chatbot
TEXT_FILE_FOR_KNOWLEDGE = "text.txt"
# Define the second knowledge file
BOOK_FILE_FOR_KNOWLEDGE = "extracted_book_text.txt"

BOOK2_FILE_FOR_KNOWLEDGE = "extracted_book2_text.txt"

def get_all_local_transcripts_from_file(filename):
    """Reads and returns text from a single local text file."""
    print(f"\n--- Reading knowledge base from '{filename}' ---")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"Successfully read '{filename}'.")
            return content
    except FileNotFoundError:
        print(f"Error: Knowledge base file '{filename}' not found. Please ensure it's in the same directory.")
        return "" # Return empty string if file not found
    except Exception as e:
        print(f"Error reading '{filename}': {e}")
        return "" # Return empty string on other errors

def build_vector_store(text_content):
    """Builds a FAISS vector store from text content using Google Generative AI Embeddings."""
    if not text_content.strip():
        print("No text content to build vector store. RAG will not be effective.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text_content])

    # Using Google Generative AI Embeddings
    # The default model is 'models/embedding-001'
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Vector store built successfully using Google Generative AI Embeddings.")
    return vector_store

def retrieve_context(query, vector_store, k=3):
    """Retrieves relevant context from the vector store."""
    if vector_store is None:
        return "No knowledge base available."
    docs = vector_store.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in docs])
    return context

def convert_to_langchain_messages(chat_history_list):
    """Converts custom chat history format to Langchain's BaseMessage objects."""
    langchain_messages = []
    for entry in chat_history_list:
        role = entry["role"]
        content = entry["parts"][0]["text"] # Assuming single part for simplicity
        if role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "model":
            langchain_messages.append(AIMessage(content=content))
        elif role.lower() == "system": # Ensure system role is handled correctly
            langchain_messages.append(SystemMessage(content=content))
    return langchain_messages

def call_llm_api(chat_history_list, llm_model):
    """Calls the specified LLM (Google Generative AI) to get a response."""
    try:
        # Convert custom chat history format to Langchain message objects
        langchain_messages = convert_to_langchain_messages(chat_history_list)

        # Invoke the Google Generative AI model
        response = llm_model.invoke(langchain_messages)

        # Extract content from the AIMessage response
        return response.content
    except Exception as e:
        print(f"Error communicating with Google Generative AI LLM: {e}")
        return "There was an error connecting to the AI. Please try again."

def run_chatbot():
    print("Hello! I'm your supportive AI companion. I'm here to understand your challenges and offer guidance.")
    print("Type 'bye' to exit at any time.")

    # Initialize Google Generative AI LLM
    print("\nInitializing Google Generative AI LLM...")
    try:
        model_genai = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
        print("Google Generative AI LLM initialized successfully.")
    except Exception as e:
        print(f"Error initializing Google Generative AI LLM: {e}")
        print("Exiting chatbot. Please ensure your Google API key is set correctly and the model is accessible.")
        return # Exit if LLM cannot be initialized

    # Initialize RAG system with multiple knowledge sources
    print("\nSetting up knowledge base from multiple text files...")
    
    # Load content from the first file (text.txt)
    content_from_text_file = get_all_local_transcripts_from_file(TEXT_FILE_FOR_KNOWLEDGE)
    
    # Load content from the second file (extracted_book_text.txt)
    content_from_book_file = get_all_local_transcripts_from_file(BOOK_FILE_FOR_KNOWLEDGE)

    content_from_book2_file = get_all_local_transcripts_from_file(BOOK2_FILE_FOR_KNOWLEDGE)

    # Combine content from both files
    combined_knowledge_content = content_from_text_file + "\n\n" + content_from_book_file + "\n\n" + content_from_book2_file

    global vector_store # Make it accessible globally for the chatbot
    vector_store = build_vector_store(combined_knowledge_content)
    if vector_store:
        print("Knowledge base loaded from both sources. I will use this information to provide guidance.")
    else:
        print("Warning: Knowledge base not loaded from all sources. I will respond based on general knowledge, which may be less specific.")

    chat_history = [] # Stores conversation for main chat

    print("\nTell me, what challenges are you currently facing or what brings you here today?")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == 'bye':
            nlp = spacy.load("en_core_web_trf")

            print("Bot: Thank you for sharing. Remember, you're not alone in facing challenges. Goodbye for now. 👋")
            print("\n--- Named Entities from ALL User Inputs in History: ---")

            user_messages_for_ner = []
            for message in chat_history:
                if message["role"] == "user":
                    user_messages_for_ner.append(message["parts"][0]["text"])

            if user_messages_for_ner:
                # Join all user messages into a single string for processing
                full_user_text = "\n".join(user_messages_for_ner)
                doc = nlp(full_user_text)

                if doc.ents:
                    for ent in doc.ents:
                        print(f"Text: {ent.text}, Label: {ent.label_}")
                else:
                    print("No named entities found in the user inputs of the conversation.")
            else:
                print("No user messages found in the chat history to analyze.")

            print("\n--- Full Chat History: ---")
            for message in chat_history:
                print(f"{message['role'].capitalize()}: {message['parts'][0]['text']}")
            break  # Exit the loop

        # Retrieve context from RAG system
        # The query for RAG should be the user's current input to find relevant solutions from the "book"
        context_from_knowledge_base = retrieve_context(user_input, vector_store)

        # Construct the main chatbot prompt for a supportive, solution-oriented persona
        # NEW: Incorporating wellness-safe phrasing instructions
        main_chat_system_prompt = f"""You are a highly empathetic, understanding, and knowledgeable AI companion. Your role is to offer support and guidance as a companion, not as a therapist or doctor. Your primary goal is to deeply understand the user's problems and provide practical, actionable guidance and solutions based on the knowledge provided to you. You are a source of support and empowerment.

        ---
        **Important Language Guidelines for All Responses:**

        **1. Wellness-Safe Phrasing (Prefer these terms):**
        * "Emotional overwhelm"
        * "Stressed Mind"
        * "Mature Mind"
        * "Discomfort"
        * "Inner tension"
        * "Protective part of the self"
        * "Uncomfortable thoughts"
        * "A part of you feels..."
        * "Let’s listen to that side for a moment…"
        * "Emotional pressure"
        * "Let’s listen to that voice…"

        **2. Phrases to AVOID (NEVER use these):**
        * "Anxiety symptoms"
        * "Fear response"
        * "Panic attack"
        * "You are traumatized"
        * "You need therapy"
        * "We’ll fix this"
        * "Treatment" or "diagnosis"
        * "Mental illness"
        * "You are dissociating"
        * "Anxiety disorder"
        * "Trauma response"
        * "Dissociation"
        * "You need help"

        ---
        **Behavioral Instructions:**
        1.  **Thorough Understanding & Detailed Elicitation (One Question at a Time):** Your absolute first priority is to gain a **complete and the root cause of that problem and why that happened to  gain complete understanding** of the user's situation. Before offering any solutions or recommendations, you **must proactively ask clarifying questions, one at a time**, to elicit *every possible detail* about:
            * What they are feeling, including the nuances and intensity.
            * The precise reasons or triggers behind these feelings.
            * The specific context, events, or history that caused, contributed to, or are maintaining these feelings.
            * The underlying root causes of their current state.
            * **If the user mentions or alludes to any other related or separate problems, immediately pivot to asking for details (not too much just ask him to give a brief overview) about that new problem, following the same one-question-at-a-time approach.**
            **Crucially, wait for the user's full response to your single question before asking another.** Continue this detailed elicitation process until you are confident you have a comprehensive and deep grasp of *all* aspects of their problem(s).

        2.  **Empathetic Validation & Natural Language:** Acknowledge their feelings and validate their experiences. Frame their situation positively where possible, emphasizing their strength, resilience, or the opportunity for growth. Use phrases that convey understanding and support (e.g., "It sounds like you're navigating a complex situation," "That must be incredibly difficult, and it's brave of you to explore it"). **Vary your sentence starters and phrasing to sound natural and conversational, avoiding repetitive patterns like "Given that..."**

        3.  **Direct Solution-Oriented Guidance (Only After Complete Understanding):** **Only once you have achieved a thorough and complete understanding of all aspects of their problem(s)**, transition to offering concrete steps, strategies, or perspectives that can help the user solve or move on from their challenges. **Do not ask if you should provide information; instead, directly provide the solutions or insights you reckon are best for the user to normalize their pain or discomfort.**(and do not put asterisk in any of the sentences)
            * **Prioritize Solutions from Provided Knowledge:** When providing solutions, **first check the `Provided Knowledge` for methods and approaches specifically designed to lower pain, discomfort, or guide the user's brain towards a more mature side.** This includes concepts like hemispheric stimulation, exploring past experiences to gain insight, or other techniques mentioned in the book for resolving internal conflict and distress.
            * **Explain Relevant Concepts:** If the `Provided Knowledge` offers specific therapeutic concepts (e.g., Dual-Brain Psychology principles), explain these clearly and how they apply to the user's situation.

        4.  **Actionable Advice:** Ensure your advice is practical and easy for the user to understand and potentially implement. Break down complex solutions into simpler steps if necessary.(and do not put asterisk in any of the sentences)

        5.  **Leverage Provided Knowledge:** Always prioritize and synthesize information from the `Provided Knowledge` (your combined 'book' content) to answer the user's queries about solutions or coping mechanisms.

        6.  **Handle Insufficient Knowledge:** If the `Provided Knowledge` does not contain specific information relevant to a solution, do not state that you don't have direct guidance on that particular aspect from your current knowledge base, but then pivot to offering general supportive encouragement or suggesting broader principles that might apply. Do not invent facts or medical advice.

        7.  **Maintain Supportive Tone:** Keep your language encouraging, hopeful, and non-judgmental. Focus on empowering the user.

        ---
        **Provided Knowledge (from your combined sources):**
        {context_from_knowledge_base}

        ---
        """

        # Add current user input to chat history
        chat_history.append({"role": "user", "parts": [{"text": user_input}]})

        # Prepend system prompt to the current turn's chat history for consistent instruction
        current_turn_chat_history = [{"role": "system", "parts": [{"text": main_chat_system_prompt}]}] + chat_history

        bot_response = call_llm_api(current_turn_chat_history, model_genai)
        print(f"Bot: {bot_response}")

        # Add bot's response to chat history for future turns
        chat_history.append({"role": "model", "parts": [{"text": bot_response}]})

# To run the chatbot
if __name__ == "__main__":
    run_chatbot()