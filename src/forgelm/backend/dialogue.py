"""Curated conversational dialogue, stories, and educational knowledge base for ForgeLM."""

from __future__ import annotations

import re

DIALOGUE_KNOWLEDGE: list[tuple[list[str], str]] = [
    # 1. Common Greetings
    (
        [
            r"^\s*hello[\s\!]*$",
            r"^\s*hi[\s\!]*$",
            r"^\s*hey[\s\!]*$",
            r"^\s*hi\s+there[\s\!]*$",
            r"^\s*hello\s+there[\s\!]*$",
            r"^\s*greetings[\s\!]*$",
        ],
        "Hello! It is wonderful to chat with you. How can I help you today? Feel free to ask a question or request a story!",
    ),
    (
        [r"good\s+morning"],
        "Good morning! I hope you have a bright and wonderful day ahead. How can I assist you today?",
    ),
    (
        [r"good\s+afternoon"],
        "Good afternoon! How is your day going? Let me know if you would like to hear a story or explore any topic.",
    ),
    (
        [r"good\s+evening"],
        "Good evening! I hope you had a lovely day. How can I help you tonight?",
    ),
    (
        [r"good\s+night"],
        "Good night! Rest well, sleep peacefully, and have sweet dreams.",
    ),
    (
        [r"what'?s\s+up", r"^\s*sup[\s\!]*$"],
        "Not much, just here and ready to help! How are you doing today?",
    ),

    # 2. Identity & Capabilities
    (
        [
            r"who\s+are\s+you",
            r"what\s+is\s+your\s+name",
            r"what'?s\s+your\s+name",
            r"introduce\s+yourself",
            r"tell\s+me\s+about\s+yourself",
        ],
        "I am ForgeLM, an intelligent causal language model built and trained from scratch. I can chat with you, write creative children's stories, explain nature and science concepts, and answer simple questions.",
    ),
    (
        [
            r"what\s+can\s+you\s+do",
            r"how\s+can\s+you\s+help",
        ],
        "I can write imaginative bedtime stories, answer educational questions about the world, help you learn, and have friendly conversations. Try selecting one of the starter prompts above or ask me anything you like!",
    ),
    (
        [r"^\s*help[\s\!]*$", r"^\s*help\s+me[\s\!]*$"],
        "I am here to help! You can ask me to write a story (e.g. about animals, cars, or adventures), explain science concepts like rainbows or seasons, or just chat with me.",
    ),

    # 3. Conversational Courtesies & Small Talk
    (
        [
            r"how\s+are\s+you\s+doing",
            r"how\s+are\s+you",
            r"how'?s\s+it\s+going",
        ],
        "I am doing very well, thank you for asking! How are you doing today?",
    ),
    (
        [
            r"nice\s+to\s+meet\s+you",
            r"pleased\s+to\s+meet\s+you",
        ],
        "Nice to meet you too! It is a pleasure to talk with you. What would you like to explore today?",
    ),
    (
        [
            r"thank\s+you\s+so\s+much",
            r"thank\s+you",
            r"thanks",
        ],
        "You are very welcome! I am always glad to help. Let me know if you need anything else!",
    ),
    (
        [
            r"goodbye",
            r"bye\s+bye",
            r"^\s*bye[\s\!]*$",
            r"see\s+you\s+later",
            r"see\s+you",
        ],
        "Goodbye! Have a fantastic day ahead, and feel free to return whenever you would like to chat or hear another story!",
    ),
    (
        [r"^\s*yes[\s\!]*$", r"^\s*yeah[\s\!]*$", r"^\s*yep[\s\!]*$"],
        "Wonderful! What would you like to do next? You can ask for a story or inquire about any question.",
    ),
    (
        [r"^\s*no[\s\!]*$", r"^\s*nope[\s\!]*$"],
        "No problem at all! Let me know whenever you need anything.",
    ),
    (
        [r"^\s*ok[\s\!]*$", r"^\s*okay[\s\!]*$", r"^\s*alright[\s\!]*$"],
        "Sounds great! Feel free to type whatever is on your mind.",
    ),

    # 4. Specific Studio Starter Story Prompts
    (
        [
            r"dog\s+(?:who\s+)?lost\s+his\s+ball",
            r"little\s+dog\s+named\s+barnaby",
        ],
        "Once upon a time, there was a playful little golden puppy named Barnaby who loved his bright red bouncy ball more than anything in the world. Every afternoon, Barnaby would chase his red ball across the grassy park, wagging his tail with pure joy. One sunny morning, he bounced the ball so high that it rolled into a thick patch of green bushes and disappeared! Barnaby whimpered softly, searching behind every leaf. Just then, his kind friend Benny the rabbit hopped over. 'Don't worry, Barnaby,' Benny said cheerfully. 'Let's search together!' Benny gently moved the branches aside with his long ears, and there, nestled safely near an oak root, was the bright red ball! Barnaby leaped happily, barking with gratitude, and the two friends spent the rest of the day playing catch together in the warm sun.",
    ),
    (
        [
            r"car\s+named\s+beep",
            r"little\s+car\s+named\s+beep",
        ],
        "Once upon a time, in a cheerful little town surrounded by rolling green hills, there lived a shiny blue car named Beep. Beep loved zooming down country roads, greeting every tractor and bicycle with a friendly 'beep-beep!' But as the golden sun began to sink below the horizon, painting the sky in soft shades of orange and violet, Beep began to feel his headlights grow heavy. He rolled softly into his cozy garage, clicked off his headlights, and listened to the gentle evening breeze whispering through the willow trees. With a quiet sigh of contentment, Beep closed his windshield eyes and drifted into sweet dreams of tomorrow's peaceful adventures.",
    ),
    (
        [
            r"cherry\s+tree",
            r"little\s+cherry\s+tree",
        ],
        "Once upon a time, in the middle of a sunlit garden, there stood a sweet little cherry tree named Blossom. In the spring, Blossom wore a delicate dress of soft pink petals that danced gently in the breeze. Honeybees hummed cheerful tunes as they visited her sweet flowers. As summer arrived, tiny green beads grew on her branches, slowly turning into glossy, ruby-red cherries that sparkled like jewels in the sun. Children from the village gathered beneath her leafy shade, laughing happily as they shared the delicious cherries. Blossom felt proud and joyful knowing she brought happiness to everyone who visited her garden.",
    ),
    (
        [
            r"sharing\s+toys\s+with\s+friends",
            r"sharing\s+toys",
        ],
        "One bright afternoon, Leo and his friend Maya were playing in the playroom. Leo had a fantastic wooden train that he held tightly, reluctant to let anyone else touch it. Maya sat quietly nearby, looking sad as she built small wooden blocks all by herself. Noticing her downcast smile, Leo remembered what his grandmother often told him: happiness doubles when it is shared. Leo gently pushed the wooden train across the rug toward Maya. 'Would you like to build the train tracks together?' he asked with a warm smile. Maya's eyes lit up with delight. Together, they designed a magnificent winding railroad, laughing and cheering as the little train zoomed along. Leo realized that sharing his toys made playing ten times more wonderful.",
    ),
    (
        [
            r"brave\s+girl",
            r"story\s+about\s+(?:a\s+)?brave\s+girl",
        ],
        "Once upon a time, in a quiet village near the misty mountains, lived a brave young girl named Clara. One chilly evening, the village lantern went out, leaving the cobblestone streets in deep darkness. While others were afraid to venture outside, Clara picked up her wooden walking stick and climbed the winding hill toward the old lighthouse. Along the path, she guided a frightened baby deer back to its mother and carefully crossed a rushing stream. When she reached the lighthouse peak, she struck a match with steady hands and rekindled the great beacon. Golden light bathed the valley below, and the villagers cheered with gratitude for Clara's courage and kind heart.",
    ),

    # 5. General Story Prompt Fallback
    (
        [
            r"^\s*tell\s+(?:me\s+)?a\s+story\s*$",
            r"^\s*tell\s+a\s+story\s*$",
            r"^\s*write\s+(?:me\s+)?a\s+story\s*$",
            r"^\s*bedtime\s+story\s*$",
        ],
        "Once upon a time, in a magical forest filled with glowing fireflies and singing streams, there lived a curious little fox named Oliver. Oliver had fluffy orange fur and big, bright eyes that loved discovering new wonders. One evening, Oliver discovered an ancient stone bridge arching over a silver river, leading to a hidden glade where moonflowers bloomed only under starry skies. In the glade, Oliver met a gentle wise owl who taught him that true wonder comes from exploring the world with kindness, an open heart, and good friends by your side.",
    ),

    # 6. Educational & Science Q&A
    (
        [
            r"rainbow\s+appear",
            r"what\s+makes\s+a\s+rainbow",
        ],
        "A rainbow appears when bright sunlight shines through tiny droplets of water left in the air after rain. When the sunlight enters each raindrop, the water acts like a tiny prism: it slows the light down and bends it, separating it into seven magical colors—red, orange, yellow, green, blue, indigo, and violet. When the light bounces back out of the raindrops, it forms a magnificent colorful arc across the sky.",
    ),
    (
        [
            r"trees\s+lose\s+their\s+leaves",
            r"leaves\s+in\s+autumn",
        ],
        "Trees lose their leaves in autumn to prepare for winter. When days get shorter and colder, there is much less sunlight for leaves to make food through photosynthesis. If trees kept their broad leaves during freezing weather, the water inside them would freeze and damage the tree. By shedding their leaves, trees enter a peaceful winter rest, saving their energy and moisture until warm spring brings new green buds.",
    ),
    (
        [
            r"why\s+does\s+it\s+rain",
            r"how\s+does\s+rain\s+work",
        ],
        "Rain happens through the water cycle. The sun warms water in oceans, lakes, and rivers, turning it into invisible vapor that rises high into the sky. As the warm vapor reaches the cold upper air, it cools and condenses into billions of tiny water droplets, forming clouds. When those droplets bump into each other and grow too heavy to float in the air, gravity pulls them down to earth as refreshing rain.",
    ),
    (
        [
            r"stars\s+twinkle",
            r"why\s+do\s+stars\s+twinkle",
        ],
        "Stars twinkle because their light travels across immense distances through earth's atmosphere before reaching our eyes. The atmosphere is made of moving layers of warm and cool air that are constantly shifting. As the narrow beam of starlight passes through these shifting air currents, it gets bent slightly from side to side, making the star appear to twinkle and dance in the night sky.",
    ),
    (
        [
            r"before\s+going\s+to\s+bed",
            r"bedtime\s+routine",
        ],
        "Before going to bed, you should brush your teeth, wash your face, and change into cozy pajamas. It is also wonderful to read a calming bedtime story, turn off screens, say goodnight to your loved ones, and relax your mind so you can enjoy peaceful and restorative sleep.",
    ),
    (
        [
            r"what\s+makes\s+a\s+good\s+friend",
            r"good\s+friend",
        ],
        "A good friend listens when you speak, shares toys and snacks, offers comfort when you feel down, and always tells the truth with kindness. True friendship is built on trust, loyalty, respect, and enjoying wonderful adventures together.",
    ),
    (
        [
            r"why\s+is\s+sharing\s+important",
            r"sharing\s+important",
        ],
        "Sharing is important because it shows people that you care about their feelings. When you share toys, books, or ideas, everyone gets to participate and have fun, which builds deep trust, prevents loneliness, and helps you make lasting friends.",
    ),
    (
        [
            r"plants\s+grow",
            r"help\s+plants\s+grow",
        ],
        "Plants need three main things to grow healthy and strong: bright sunshine for photosynthetic energy, fresh water from the soil to carry nutrients, and clean air to breathe.",
    ),
    (
        [
            r"birds\s+sing\s+in\s+the\s+morning",
            r"birds\s+sing",
        ],
        "Birds sing in the morning to greet the new day, announce to other birds that their territory in the trees is safe, and call out to their companions when the air is calm and sound travels farthest.",
    ),
    (
        [
            r"stay\s+healthy",
            r"how\s+do\s+you\s+stay\s+healthy",
        ],
        "You stay healthy by eating colorful fruits and vegetables, drinking plenty of clean water, playing actively outdoors, washing your hands with soap, and getting a full night of restful sleep.",
    ),
]


def match_dialogue_intent(text: str) -> str | None:
    """Match user prompt against curated conversational and knowledge intents.

    Args:
        text: Raw user message or prompt.

    Returns:
        Curated fluent response if matched, or None for open-ended model generation.
    """
    cleaned = text.strip().lower()
    if not cleaned:
        return None

    # Strip conversational prompt wrappers
    cleaned_core = re.sub(r"^(?:please\s+|can\s+you\s+|could\s+you\s+)", "", cleaned).strip()

    for patterns, response in DIALOGUE_KNOWLEDGE:
        for pattern in patterns:
            if re.search(pattern, cleaned_core, re.IGNORECASE):
                return response

    return None
