
"""
Week 9 - Chatbot Development
Interactive local AI chatbot using Ollama and Python.

Run:
    python chatbot.py
    python chatbot.py --model llama3.2:3b
    python chatbot.py --domain Education
"""

import argparse
import csv
import logging
import uuid
from datetime import datetime
from pathlib import Path

import ollama


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATASET_DIR = ROOT / "dataset"
NOTEBOOK_DIR = ROOT / "notebook"

HISTORY_FILE = DATASET_DIR / "chat_history.csv"
SAMPLE_FILE = DATASET_DIR / "sample_questions.csv"
LOG_FILE = DATASET_DIR / "chatbot_logs.txt"
RESPONSES_FILE = DATASET_DIR / "responses.csv"

CSV_COLUMNS = [
    "timestamp",
    "session_id",
    "turn",
    "user_message",
    "assistant_response",
    "model",
    "domain",
]


# ============================================================
# 2. CHATBOT CONFIGURATION
# ============================================================

SYSTEM_PROMPT = """
You are a helpful, friendly, and responsible {domain} assistant.

Instructions:
- Answer questions clearly and accurately.
- Explain difficult concepts step by step.
- Use examples when they are helpful.
- Ask for clarification if a question is unclear.
- Do not invent facts or pretend to know something you do not know.
- For coding questions, provide readable code and explanations.
- Keep responses appropriate for students.
"""


# ============================================================
# 3. SAMPLE QUESTIONS
# ============================================================

SAMPLE_QUESTIONS = [
    ("Factual", "What is Python?"),
    ("Factual", "What is a variable in Python?"),
    ("Factual", "What is a function?"),
    ("Factual", "What is a database?"),
    ("Factual", "What is an algorithm?"),
    ("Factual", "What is the difference between RAM and storage?"),
    ("Factual", "What does CPU stand for?"),
    ("Factual", "What is artificial intelligence?"),
    ("Reasoning", "I have 3 apples and receive 2 more, then give away 1. How many remain?"),
    ("Reasoning", "Why should a program validate user input?"),
    ("Reasoning", "How would you debug a program that crashes for certain inputs?"),
    ("Reasoning", "What is the average of 70, 80, 90, and 60?"),
    ("Reasoning", "Explain one advantage and disadvantage of CSV files."),
    ("Reasoning", "How can you investigate why a program is running slowly?"),
    ("Coding", "Write a Python function that returns the square of a number."),
    ("Coding", "Write a Python function that checks whether a number is even."),
    ("Coding", "Write a Python loop that prints numbers from 1 to 5."),
    ("Coding", "Explain Python try and except with an example."),
    ("Coding", "Show how to read a CSV file using Python."),
    ("Coding", "Write a Python program to calculate the sum of two numbers."),
    ("General", "Hello! What can you help me with?"),
    ("General", "Give me three tips for learning programming."),
    ("General", "Explain artificial intelligence to a beginner."),
    ("General", "How can I improve my problem-solving skills?"),
    ("Context", "My favourite programming language is Python."),
    ("Context", "What is my favourite programming language?"),
]


# ============================================================
# 4. CREATE REQUIRED DIRECTORIES AND FILES
# ============================================================

def prepare_files():
    """Create the project folders and required dataset files."""

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)

    # Create CSV files with headers if they do not exist.
    for file_path in (HISTORY_FILE, RESPONSES_FILE):
        if not file_path.exists():
            with file_path.open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.writer(file)
                writer.writerow(CSV_COLUMNS)

    # Create sample questions only if the file is missing.
    if not SAMPLE_FILE.exists():
        with SAMPLE_FILE.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["category", "question"])
            writer.writerows(SAMPLE_QUESTIONS)

    # Create the log file if necessary.
    LOG_FILE.touch(exist_ok=True)

    # Configure application logging.
    logging.basicConfig(
        filename=str(LOG_FILE),
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        encoding="utf-8",
    )


# ============================================================
# 5. SAVE CONVERSATION HISTORY
# ============================================================

def save_turn(
    session_id,
    turn,
    user_message,
    assistant_response,
    model,
    domain,
):
    """Save one successful conversation turn to both CSV files."""

    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    row = [
        timestamp,
        session_id,
        turn,
        user_message,
        assistant_response,
        model,
        domain,
    ]

    try:
        for file_path in (HISTORY_FILE, RESPONSES_FILE):
            with file_path.open(
                "a", newline="", encoding="utf-8"
            ) as file:
                writer = csv.writer(file)
                writer.writerow(row)

        logging.info(
            "Conversation saved | session=%s | turn=%s",
            session_id,
            turn,
        )

    except OSError:
        logging.exception("Could not save conversation to CSV.")
        print(
            "WARNING: The response was generated, "
            "but saving the CSV record failed."
        )


# ============================================================
# 6. CHECK OLLAMA AND THE SELECTED MODEL
# ============================================================

def check_ollama(model):
    """Verify that Ollama is available and the model is installed."""

    try:
        ollama.list()

    except Exception as exc:
        raise RuntimeError(
            "Cannot connect to Ollama. Open the Ollama application "
            "and try again. You can also run 'ollama list' "
            "in your terminal."
        ) from exc

    try:
        ollama.show(model)

    except Exception as exc:
        raise RuntimeError(
            f"Model '{model}' is unavailable. "
            f"Download it using: ollama pull {model}"
        ) from exc


# ============================================================
# 7. CHATBOT MAIN LOOP
# ============================================================

def chat(model="llama3.2:3b", domain="Education"):
    """Run an interactive chatbot with conversation context."""

    prepare_files()

    print("=" * 65)
    print("       WEEK 9 - INTERACTIVE AI CHATBOT")
    print("=" * 65)
    print(f"Domain : {domain}")
    print(f"Model  : {model}")
    print()
    print("Commands:")
    print("  /help   - Display available commands")
    print("  /clear  - Clear the current conversation context")
    print("  /exit   - Exit the chatbot")
    print()
    print("Conversation records are saved automatically.")
    print("=" * 65)

    check_ollama(model)

    # A unique ID distinguishes different chatbot sessions.
    session_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:6]
    )

    # Keep messages in memory to support context-aware answers.
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT.format(domain=domain),
    }

    messages = [system_message]
    turn = 0

    logging.info(
        "Chatbot session started | session=%s | model=%s | domain=%s",
        session_id,
        model,
        domain,
    )

    print("\nChatbot is ready! Ask your first question.")

    while True:
        try:
            user_message = input("\nYou: ").strip()

        except (EOFError, KeyboardInterrupt):
            print("\n\nChatbot session ended.")
            break

        if not user_message:
            print("Please enter a question.")
            continue

        command = user_message.lower()

        # Help command
        if command == "/help":
            print("\nAvailable commands:")
            print("/help  - Show commands")
            print("/clear - Reset conversation context")
            print("/exit  - Exit the chatbot")
            continue

        # Clear conversation context
        if command == "/clear":
            messages = [system_message]
            print(
                "Conversation context cleared. "
                "Previously saved records have been preserved."
            )
            continue

        # Exit command
        if command in {"/exit", "exit", "quit"}:
            print("\nGoodbye! Your conversation has been saved.")
            break

        # Add user message to the current context.
        messages.append({
            "role": "user",
            "content": user_message,
        })

        try:
            result = ollama.chat(
                model=model,
                messages=messages,
            )

            answer = result["message"]["content"].strip()

            if not answer:
                raise ValueError("The model returned an empty response.")

            # Add assistant response to context.
            messages.append({
                "role": "assistant",
                "content": answer,
            })

            turn += 1

            # Save successful question and answer.
            save_turn(
                session_id=session_id,
                turn=turn,
                user_message=user_message,
                assistant_response=answer,
                model=model,
                domain=domain,
            )

            print(f"\nBot: {answer}")

        except Exception as exc:
            # Remove the failed user message from context.
            messages.pop()

            logging.exception("Failed to generate chatbot response.")

            print(f"\nERROR: {exc}")
            print(
                "Please check that Ollama is running, "
                "the model is installed, and enough memory is available."
            )

    logging.info(
        "Chatbot session ended | session=%s | successful_turns=%s",
        session_id,
        turn,
    )


# ============================================================
# 8. PROGRAM ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Interactive local AI chatbot using Ollama."
    )

    parser.add_argument(
        "--model",
        default="llama3.2:3b",
        help="Name of the installed Ollama model",
    )

    parser.add_argument(
        "--domain",
        default="Education",
        help="Subject domain for the chatbot",
    )

    args = parser.parse_args()

    try:
        chat(
            model=args.model,
            domain=args.domain,
        )

    except RuntimeError as exc:
        prepare_files()
        logging.error("Startup failed: %s", exc)
        print(f"\nSETUP ERROR: {exc}")


if __name__ == "__main__":
    main()
