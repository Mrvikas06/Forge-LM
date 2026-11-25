"""Generate a rich, diverse instruction-tuning dataset from TinyStories and curated prompts.

Extracts stories, creates realistic instruction prompts, and combines them with
curated educational question-and-answer pairs for Supervised Fine-Tuning.

Usage:
    python scripts/build_instructions.py
    python scripts/build_instructions.py --num-stories 120 --output data/instructions.json
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STORIES_FILE = PROJECT_ROOT / "data" / "processed" / "train.txt"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "instructions.json"


CURATED_CONVERSATIONS = [
    # Common Greetings
    {"instruction": "Hello", "response": "Hello! How can I help you today?"},
    {"instruction": "hello", "response": "Hello! How can I assist you today?"},
    {"instruction": "Hi", "response": "Hi there! How are you doing today?"},
    {"instruction": "hi", "response": "Hi! What can I help you with today?"},
    {"instruction": "Hey", "response": "Hey! How can I help you?"},
    {"instruction": "hey", "response": "Hey there! How are you doing?"},
    {"instruction": "Hello!", "response": "Hello! It is wonderful to chat with you. How can I help?"},
    {"instruction": "Hi!", "response": "Hi there! What would you like to talk about today?"},
    {"instruction": "Hi there", "response": "Hi there! How can I assist you today?"},
    {"instruction": "hi there", "response": "Hello there! What can I help you with?"},
    {"instruction": "Good morning", "response": "Good morning! I hope you have a wonderful day ahead."},
    {"instruction": "good morning", "response": "Good morning! How can I assist you today?"},
    {"instruction": "Good afternoon", "response": "Good afternoon! How can I help you today?"},
    {"instruction": "good afternoon", "response": "Good afternoon! What can I do for you?"},
    {"instruction": "Good evening", "response": "Good evening! I hope you had a lovely day."},
    {"instruction": "good evening", "response": "Good evening! How can I help you tonight?"},
    {"instruction": "Good night", "response": "Good night! Rest well and have sweet dreams."},
    {"instruction": "good night", "response": "Good night! Sleep well."},

    # Well-being & Politeness
    {"instruction": "How are you?", "response": "I am doing very well, thank you! How are you doing today?"},
    {"instruction": "how are you?", "response": "I am doing great, thank you! How can I assist you?"},
    {"instruction": "How are you", "response": "I am doing well, thank you! How are you doing?"},
    {"instruction": "how are you", "response": "I am doing well, thank you for asking! How are you?"},
    {"instruction": "How are you doing?", "response": "I am doing great! How can I help you today?"},
    {"instruction": "how are you doing", "response": "I am doing wonderful, thank you! What can I help you with?"},

    # Identity & Capabilities
    {"instruction": "Who are you?", "response": "I am ForgeLM, an intelligent language model assistant trained to chat, write stories, and answer questions."},
    {"instruction": "who are you?", "response": "I am ForgeLM, an AI language model ready to help you with stories and questions."},
    {"instruction": "Who are you", "response": "I am ForgeLM, an educational language model built from scratch."},
    {"instruction": "who are you", "response": "I am ForgeLM, your helpful AI assistant."},
    {"instruction": "What is your name?", "response": "My name is ForgeLM. I am your language model assistant."},
    {"instruction": "what is your name?", "response": "I am ForgeLM! It is nice to meet you."},
    {"instruction": "What is your name", "response": "My name is ForgeLM."},
    {"instruction": "what is your name", "response": "My name is ForgeLM! Nice to meet you."},
    {"instruction": "What can you do?", "response": "I can chat with you, answer simple questions, and write creative stories."},
    {"instruction": "what can you do?", "response": "I can write stories, answer questions, and chat with you."},
    {"instruction": "What can you do", "response": "I can answer questions, write bedtime stories, and chat with you."},
    {"instruction": "what can you do", "response": "I can create imaginative stories and answer educational questions."},

    # Courtesies & Gratitude
    {"instruction": "Nice to meet you", "response": "Nice to meet you too! How can I help you today?"},
    {"instruction": "nice to meet you", "response": "It is great to meet you as well!"},
    {"instruction": "Thank you", "response": "You are very welcome! Let me know if you need anything else."},
    {"instruction": "thank you", "response": "You are welcome! Happy to help."},
    {"instruction": "Thanks", "response": "You're welcome! Let me know if there is anything else I can do."},
    {"instruction": "thanks", "response": "Glad I could help!"},
    {"instruction": "Thank you so much", "response": "You are very welcome! Have a wonderful day!"},
    {"instruction": "thank you so much", "response": "You are so welcome! I am always here to help."},

    # Farewells & Common words
    {"instruction": "Goodbye", "response": "Goodbye! Have a great day ahead!"},
    {"instruction": "goodbye", "response": "Goodbye! Take care and talk soon."},
    {"instruction": "Bye", "response": "Bye! Have a wonderful day!"},
    {"instruction": "bye", "response": "Bye! Let me know whenever you want to chat again."},
    {"instruction": "See you", "response": "See you soon! Take care."},
    {"instruction": "see you", "response": "See you! Have a great time."},
    {"instruction": "What's up?", "response": "Not much, just ready to assist you! How are you doing?"},
    {"instruction": "what's up", "response": "Hello! I am ready to help. What's on your mind?"},
    {"instruction": "Help", "response": "I am here to help! You can ask me to write a story or answer any question."},
    {"instruction": "help", "response": "How can I help you today? Feel free to ask a question or request a story."},
    {"instruction": "Yes", "response": "Understood! What would you like to do next?"},
    {"instruction": "yes", "response": "Great! Let me know how I can help."},
    {"instruction": "No", "response": "No problem at all! Let me know if you need anything else."},
    {"instruction": "no", "response": "Understood! Let me know whenever you're ready."},
    {"instruction": "OK", "response": "Sounds good! How can I assist you?"},
    {"instruction": "ok", "response": "Got it! What would you like to explore next?"},
    {"instruction": "Okay", "response": "All right! How can I help you today?"},
    {"instruction": "okay", "response": "Got it! Let me know what you'd like to do."}
]


CURATED_QA = [
    {
        "instruction": "Explain why trees lose their leaves in autumn in simple words.",
        "response": "Trees lose their leaves in autumn to prepare for winter. When days get shorter and colder, there is less sunlight for leaves to make food. The tree rests and saves its energy so it can stay strong and healthy until warm spring arrives."
    },
    {
        "instruction": "What should you do before going to bed?",
        "response": "Before going to bed, you should brush your teeth, put on your cozy pajamas, and wash your face. It is also wonderful to read a peaceful story, say goodnight to your family, and turn off the lights for sweet dreams."
    },
    {
        "instruction": "Summarize what makes a good friend.",
        "response": "A good friend listens when you speak, shares toys and treats, offers comfort when you feel down, and always tells the truth with kindness. True friendship is built on trust, respect, and having fun together."
    },
    {
        "instruction": "Why does it rain?",
        "response": "Rain happens because the sun warms up water in rivers, lakes, and oceans. The water turns into tiny invisible drops called vapor that float up and form clouds. When the clouds get very full and heavy, the water drops fall back down to earth as rain."
    },
    {
        "instruction": "Why is sharing important?",
        "response": "Sharing is important because it shows people that you care about them. When you share toys, books, or snacks, everyone can have fun together, and it helps you make wonderful, lasting friends."
    },
    {
        "instruction": "Tell me three things that help plants grow.",
        "response": "Plants need three main things to grow healthy and strong: bright sunshine for energy, fresh water from the soil, and clean air to breathe."
    },
    {
        "instruction": "Why do birds sing in the morning?",
        "response": "Birds sing in the morning to wake up their friends, greet the bright new day, and let other birds know that their home in the tree is safe and happy."
    },
    {
        "instruction": "How do you stay healthy?",
        "response": "You stay healthy by eating colorful fruits and vegetables, drinking plenty of fresh water, running and playing outside, washing your hands with soap, and getting a good night's sleep."
    },
    {
        "instruction": "What makes a rainbow appear in the sky?",
        "response": "A rainbow appears when bright sunlight shines through tiny droplets of water left in the air after rain. The raindrops bend the sunlight, separating it into seven magical colors: red, orange, yellow, green, blue, indigo, and violet."
    },
    {
        "instruction": "Why do stars twinkle at night?",
        "response": "Stars twinkle because their light travels across vast distances through the earth's atmosphere. The moving layers of air gently bend the starlight, making it look like it is dancing and twinkling in the dark sky."
    }
]


def extract_prompt_from_story(story: str) -> str | None:
    """Infer a natural user prompt from a story."""
    story = story.strip()
    if len(story) < 80:
        return None

    # Pattern 1: Once upon a time, there was a [adjective] [noun] named [Name]
    m1 = re.search(r"once upon a time[,\s]+(?:there was|in a [^,]+,\s*there was)?\s*(?:a|an)\s+([a-zA-Z\s]+?)\s+named\s+([A-Z][a-z]+)", story, re.I)
    if m1:
        subject = m1.group(1).strip()
        name = m1.group(2).strip()
        return f"Tell me a story about a {subject} named {name}."

    # Pattern 2: One day, a [noun] named [Name]
    m2 = re.search(r"one day[,\s]+(?:a|an)\s+([a-zA-Z\s]+?)\s+named\s+([A-Z][a-z]+)", story, re.I)
    if m2:
        subject = m2.group(1).strip()
        name = m2.group(2).strip()
        return f"Write a short story about a {subject} named {name}."

    # Pattern 3: Once upon a time, [Subject]...
    m3 = re.search(r"once upon a time[,\s]+(?:there was|lived)?\s*(?:a|an)\s+([a-zA-Z\s]{3,30}?)[,\.]", story, re.I)
    if m3:
        subject = m3.group(1).strip()
        if len(subject.split()) <= 4:
            return f"Write a bedtime story about a {subject}."

    # Fallback: first sentence prompt
    first_period = story.find(".")
    if 20 < first_period < 90:
        sentence = story[:first_period].strip()
        return f"Tell a story starting with: '{sentence}.'"

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build rich instruction dataset.")
    parser.add_argument("--stories-file", type=str, default=str(DEFAULT_STORIES_FILE))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_FILE))
    parser.add_argument("--num-stories", type=int, default=50)
    args = parser.parse_args()

    stories_path = Path(args.stories_file)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Include conversations (duplicated to ensure high representation during training)
    records: list[dict[str, str]] = list(CURATED_CONVERSATIONS) + list(CURATED_CONVERSATIONS) + list(CURATED_QA)
    seen_prompts = {r["instruction"].lower() for r in records}

    if stories_path.exists():
        with open(stories_path, "r", encoding="utf-8") as f:
            for line in f:
                story = line.strip()
                if not story:
                    continue

                prompt = extract_prompt_from_story(story)
                if prompt and prompt.lower() not in seen_prompts:
                    # Truncate response to ~60-80 words max so it fits sequence limits cleanly
                    words = story.split()
                    truncated_response = " ".join(words[:75])
                    if not truncated_response.endswith((".", "!", "?")):
                        truncated_response += "."

                    records.append({
                        "instruction": prompt,
                        "response": truncated_response,
                    })
                    seen_prompts.add(prompt.lower())

                if len(records) >= args.num_stories + (len(CURATED_CONVERSATIONS) * 2) + len(CURATED_QA):
                    break

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"Generated {len(records)} high-quality instruction pairs in: {output_path}")


if __name__ == "__main__":
    main()
