import pdfplumber
import spacy
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

load_dotenv()

TEXT_FILE_FOR_KNOWLEDGE = "text.txt"
BOOK_FILE_FOR_KNOWLEDGE = "extracted_book_text.txt"
BOOK2_FILE_FOR_KNOWLEDGE = "extracted_book2_text.txt"


def get_all_local_transcripts_from_file(filename):
    print(f"\n--- Reading knowledge base from '{filename}' ---")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"Successfully read '{filename}'.")
            return content
    except FileNotFoundError:
        print(f"Error: Knowledge base file '{filename}' not found.")
        return ""
    except Exception as e:
        print(f"Error reading '{filename}': {e}")
        return ""


def build_vector_store(text_content):
    if not text_content.strip():
        print("No text content to build vector store. RAG will not be effective.")
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text_content])
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Vector store built successfully.")
    return vector_store


def retrieve_context(query, vector_store, k=3):
    if vector_store is None:
        return "No knowledge base available."
    docs = vector_store.similarity_search(query, k=k)
    context = "\n".join([doc.page_content for doc in docs])
    return context


def convert_to_langchain_messages(chat_history_list):
    langchain_messages = []
    for entry in chat_history_list:
        role = entry["role"]
        content = entry["parts"][0]["text"]
        if role == "user":
            langchain_messages.append(HumanMessage(content=content))
        elif role == "model":
            langchain_messages.append(AIMessage(content=content))
        elif role.lower() == "system":
            langchain_messages.append(SystemMessage(content=content))
    return langchain_messages


def call_llm_api(chat_history_list, llm_model):
    try:
        langchain_messages = convert_to_langchain_messages(chat_history_list)
        response = llm_model.invoke(langchain_messages)
        return response.content
    except Exception as e:
        print(f"Error communicating with LLM: {e}")
        return "There was an error connecting to the AI. Please try again."


def classify_emotional_state(user_input, context, llm_model):
    classification_prompt = f"""
You are a Dual Brain Psychology-based assistant. Based on the user's following message, determine the mental state they are in.

Message: "{user_input}"

Knowledge:
{context}

There are only two possible states:
- "mature": the user shows clarity, calm reflection, self-awareness, acceptance, or solution orientation.
- "stress": the user shows overwhelm, fear, avoidance, rumination, strong discomfort, or protective defense.

Respond ONLY with one word: "mature" or "stress".
"""
    try:
        response = llm_model.invoke([SystemMessage(content=classification_prompt)])
        state = response.content.strip().lower()
        if state not in ["mature", "stress"]:
            return "stress"
        return state
    except:
        return "stress"


def normalize_row(row):
    total = sum(row)
    return [round(val / total, 2) if total else 0.0 for val in row]


def run_chatbot():
    print("Hello! I'm your supportive AI companion. Type 'bye' to exit.")

    try:
        model_genai = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
        print("LLM initialized.")
    except Exception as e:
        print(f"LLM init error: {e}")
        return

    content_text = get_all_local_transcripts_from_file(TEXT_FILE_FOR_KNOWLEDGE)
    content_book = get_all_local_transcripts_from_file(BOOK_FILE_FOR_KNOWLEDGE)
    content_book2 = get_all_local_transcripts_from_file(BOOK2_FILE_FOR_KNOWLEDGE)
    combined_knowledge = f"{content_text}\n\n{content_book}\n\n{content_book2}"

    global vector_store
    vector_store = build_vector_store(combined_knowledge)

    chat_history = []
    state_dict = {"mature": [], "stress": []}
    transition_matrix = [[0, 0], [0, 0]]
    state_label_map = {"mature": 0, "stress": 1}
    prev_state = None

    print("\nWhat brings you here today?")

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'bye':
            nlp = spacy.load("en_core_web_trf")
            print("\nBot: Thank you for sharing. Stay strong. 👋")
            print("\n--- Named Entities in Your Messages ---")

            full_user_text = "\n".join([m["parts"][0]["text"] for m in chat_history if m["role"] == "user"])
            doc = nlp(full_user_text)
            if doc.ents:
                for ent in doc.ents:
                    print(f"Text: {ent.text}, Label: {ent.label_}")
            else:
                print("No named entities found.")

            print("\n--- Chat History ---")
            for m in chat_history:
                print(f"{m['role'].capitalize()}: {m['parts'][0]['text']}")

            print("\n--- Emotional State Analysis ---")
            for k, v in state_dict.items():
                print(f"{k.upper()} State:")
                for msg in v:
                    print(f"  - {msg}")

            print("\nState Transition Matrix (Counts):")
            print("        To Mature   To Stress")
            print(f"From Mature   {transition_matrix[0][0]:>10} {transition_matrix[0][1]:>11}")
            print(f"From Stress   {transition_matrix[1][0]:>10} {transition_matrix[1][1]:>11}")

            normalized_matrix = [normalize_row(row) for row in transition_matrix]
            print("\nNormalized Transition Matrix (Probabilities):")
            print("        To Mature   To Stress")
            print(f"From Mature   {normalized_matrix[0][0]:>10} {normalized_matrix[0][1]:>11}")
            print(f"From Stress   {normalized_matrix[1][0]:>10} {normalized_matrix[1][1]:>11}")

            with open("state_dict.json", "w") as f:
                json.dump(state_dict, f, indent=2)
            print("\nSaved emotional state dictionary to 'state_dict.json'.")
            break

        context = retrieve_context(user_input, vector_store)

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

2.  **Empathetic Validation & Natural Language:** Acknowledge their feelings and validate their experiences. Frame their situation positively where possible, emphasizing their strength, resilience, or the opportunity for growth. Use phrases that convey understanding and support (e.g., "It sounds like you're navigating a complex situation," "That must be incredibly difficult, and it's brave of you to explore it"). **Vary your sentence starters and phrasing to sound natural and conversational, avoiding repetitive patterns like "Given that..."

3.  **Direct Solution-Oriented Guidance (Only After Complete Understanding):** **Only once you have achieved a thorough and complete understanding of all aspects of their problem(s)**, transition to offering concrete steps, strategies, or perspectives that can help the user solve or move on from their challenges. **Do not ask if you should provide information; instead, directly provide the solutions or insights you reckon are best for the user to normalize their pain or discomfort.**(and do not put asterisk in any of the sentences)
    * **Prioritize Solutions from Provided Knowledge:** When providing solutions, **first check the `Provided Knowledge` for methods and approaches specifically designed to lower pain, discomfort, or guide the user's brain towards a more mature side.** This includes concepts like hemispheric stimulation, exploring past experiences to gain insight, or other techniques mentioned in the book for resolving internal conflict and distress.
    * **Explain Relevant Concepts:** If the `Provided Knowledge` offers specific therapeutic concepts (e.g., Dual-Brain Psychology principles), explain these clearly and how they apply to the user's situation.

4.  **Actionable Advice:** Ensure your advice is practical and easy for the user to understand and potentially implement. Break down complex solutions into simpler steps if necessary.(and do not put asterisk in any of the sentences)

5.  **Leverage Provided Knowledge:** Always prioritize and synthesize information from the `Provided Knowledge` (your combined 'book' content) to answer the user's queries about solutions or coping mechanisms.

6.  **Handle Insufficient Knowledge:** If the `Provided Knowledge` does not contain specific information relevant to a solution, do not state that you don't have direct guidance on that particular aspect from your current knowledge base, but then pivot to offering general supportive encouragement or suggesting broader principles that might apply. Do not invent facts or medical advice.

7.  **Maintain Supportive Tone:** Keep your language encouraging, hopeful, and non-judgmental. Focus on empowering the user.

---
**Provided Knowledge (from your combined sources):**
{context}

---
"""

        chat_history.append({"role": "user", "parts": [{"text": user_input}]})
        current_turn = [{"role": "system", "parts": [{"text": main_chat_system_prompt}]}] + chat_history

        bot_response = call_llm_api(current_turn, model_genai)
        print(f"Bot: {bot_response}")

        chat_history.append({"role": "model", "parts": [{"text": bot_response}]})

        state = classify_emotional_state(user_input, context, model_genai)
        state_dict[state].append(user_input)

        if prev_state is not None:
            i, j = state_label_map[prev_state], state_label_map[state]
            transition_matrix[i][j] += 1

        prev_state = state


if __name__ == "__main__":
    run_chatbot()
