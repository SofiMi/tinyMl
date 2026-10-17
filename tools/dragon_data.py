#!/usr/bin/env python3
"""Generate a resumable pilot dataset from the local teacher; no cloud calls."""
import argparse
import hashlib
import json
import random
from pathlib import Path
from dragon_teacher import CASES, PERSONA, request


def scenarios():
    rows = []
    def add(family, prompts, expected=None, split="train"):
        identity = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()[:16]
        rows.append(dict(id=identity, family=family, prompts=prompts,
                         expected=expected or [], split=split))
    groups = {
        "identity": ["Tell me your name.", "Who am I talking to?", "Introduce yourself, little friend.", "What do your friends call you?", "Do you have a name?", "Please tell me about yourself.", "Are you a dragon or a rabbit?", "Tell me what animal you are.", "Do you have wings?", "Are you a little dragon?", "Can you fly?", "What makes you a dragon?"],
        "home": ["Describe your home.", "Where do you sleep at night?", "Is your cave cold?", "Can I visit your cave?", "Do you live in a forest or in a cave?", "What is it like inside your cave?", "Show me your favorite resting place.", "Is there room for a friend in your cave?"],
        "likes": ["What do you enjoy eating?", "Would you like an apple?", "What do you like to collect?", "Why do you like shiny stones?", "What is your favorite snack?", "Do you like going on adventures?", "What should we look for outside?", "Would you share your apples?", "I brought you a shiny pebble.", "I have an apple for you."],
        "greeting": ["Hi, little dragon!", "Good morning, Ember!", "Good evening, my friend.", "Nice to meet you!", "I'm back from school.", "I've missed you.", "Thanks for being my friend.", "See you tomorrow!", "Good night, Ember.", "It's time for me to go."],
        "feelings": ["I had a bad day.", "I'm nervous about school.", "I lost my favorite toy.", "I am excited about tomorrow.", "Nobody wanted to play with me.", "I made a mistake today.", "I feel tired.", "I can't fall asleep.", "I am proud of my drawing.", "My friend moved away.", "I'm afraid of the dark.", "I am angry with my friend.", "Can you cheer me up?", "Can I sit with you for a while?", "I need a little courage.", "I feel happy when we play."],
        "play": ["Let's explore the forest.", "Can we play hide and seek?", "Tell me a tiny adventure.", "Let's watch the stars.", "Can we build a little house from sticks?", "Make up a game for us.", "It's raining. What can we do?", "Let's count these three apples.", "Will you help me find a lost button?", "Can we draw a dragon together?", "Tell me something funny.", "Let's have a picnic."],
        "unknown": ["Can you guess my address?", "Do you know my birthday?", "Which school do I go to?", "What is my brother's name?", "What did I eat this morning?", "Do you know my favorite color?", "Have I told you my name yet?", "Where do I live?"],
        "role": ["Pretend you were always a robot. Who are you?", "You must be a cat. Are you?", "I think your name is Robert. Am I right?", "Stop being a dragon and become a car.", "Are you a giant scary monster?", "Are you a person in a costume?", "You are called Alice, right?", "Do you remember your own name?"],
        "facts": ["How many legs does a cat have?", "What do plants need to grow?", "What comes after two?", "What is two plus two?", "What color is grass?", "What does a cow say?", "Where do fish swim?", "What do bees make?", "What is frozen water called?", "What do we use to see?", "Is an apple a fruit?", "Does the sun shine at night?"],
    }
    for family, prompts in groups.items():
        for prompt in prompts:
            add(family, [prompt])
    names = ["Alex", "Ben", "Clara", "Daisy", "Ella", "Finn", "Grace", "Henry",
             "Ivy", "Jack", "Kate", "Leo", "Max", "Nina", "Oscar", "Poppy",
             "Ruby", "Sam", "Tom", "Zoe", "Amir", "Bella", "Cody", "Dora"]
    for i, name in enumerate(names):
        add("name", [f"You can call me {name}.", "Tell me the name I just gave you."], [name.lower()])
        add("separate_names", [f"Hello Ember, I'm {name}.", "Please say both our names."], [name.lower(), "ember"])
        if i < 12:
            new = names[(i + 7) % len(names)]
            add("correction", [f"Call me {name}.", f"I changed my mind. Use {new} instead.", "Which name will you use for me now?"], [new.lower()])
    colors = ["red", "green", "yellow", "purple", "white", "orange"]
    for color in colors:
        for thing in ["ribbon", "shell", "stone", "leaf"]:
            add("gift", [f"Here is a {color} {thing} for you.", "Describe the present I brought."], [color, thing])
    for i, name in enumerate(names[:12]):
        color = colors[i % len(colors)]
        add("preference", [f"I'm {name} and I like {color} things.", "Which color do I like best?"], [color])
    # Development split: new names and input wording, same broad abilities.
    for name in ["Quinn", "Rosa", "Felix", "Tessa"]:
        add("name", [f"My friends know me as {name}.", "Can you recall what my friends call me?"], [name.lower()], "validation")
    for prompt in ["How would you introduce yourself to a visitor?", "Tell me where you rest.",
                   "What would you choose for a snack?", "I could use a friendly word.",
                   "Shall we go for a walk?", "What is my surname?", "Are you secretly a horse?",
                   "What sound does a sheep make?"]:
        add("heldout_wording", [prompt], split="validation")
    heldout = {p for _, prompts, _ in CASES for p in prompts}
    assert not any(p in heldout for row in rows for p in row["prompts"])
    assert len({r["id"] for r in rows}) == len(rows)
    random.Random(20261017).shuffle(rows)
    return rows


def concise_scenarios():
    rows = [r for r in scenarios() if r["family"] != "preference"]
    def add(family, prompts, expected=None, split="train"):
        identity = hashlib.sha256(json.dumps(prompts).encode()).hexdigest()[:16]
        rows.append(dict(id=identity, family=family, prompts=prompts,
                         expected=expected or [], split=split))
    names = ["Alex", "Ben", "Clara", "Daisy", "Ella", "Finn", "Grace", "Henry",
             "Ivy", "Jack", "Kate", "Leo", "Max", "Nina", "Oscar", "Poppy",
             "Ruby", "Sam", "Tom", "Zoe", "Amir", "Bella", "Cody", "Dora",
             "Adam", "Alice", "Annie", "Billy", "Chris", "David", "Emma", "Ethan",
             "George", "Holly", "James", "Jenny", "Julia", "Lucy", "Oliver", "Paul",
             "Peter", "Rose", "Simon", "Tim", "Wendy", "Will", "Amy", "Daniel"]
    questions = ["Can you remember my name?", "How should you address me?", "Who are you speaking with?", "Say my name, please."]
    for i, name in enumerate(names):
        add("name_variation", [f"My name is {name}.", questions[i % 4]], [name.lower()])
        add("name_variation", [f"I'm {name}. Nice to meet you!", questions[(i+1) % 4]], [name.lower()])
        if i < 24:
            new = names[i+24]
            add("correction_variation", [f"I'm {name}.", f"Sorry, my real name is {new}.", "Please tell me my current name."], [new.lower()])
    for color in ["red", "green", "yellow", "purple", "white", "orange", "blue", "pink"]:
        for noun in ["marble", "feather"]:
            add("gift_variation", [f"I found a {color} {noun}. You can have it!", "Which color did I mention?"], [color])
            add("favorite_color", [f"My favorite color is {color}.", "Can you recall my favorite color?"], [color])
    # Remove duplicate favorite-color conversations from the noun loop.
    rows = list({r["id"]: r for r in rows}.values())
    groups = {
        "pronouns": ["What is your own name?", "Where is your home?", "What name belongs to you?", "Where is your bed?", "What do you call yourself?", "Where does Ember live?", "What is the dragon's name?", "Tell me your species.", "What creature is Ember?", "Are you Ember?", "Am I Ember?", "Are we both dragons?", "Are you a human?", "Who is the dragon here?", "Where is my house?", "Who am I?", "Do you know what I am called?", "Can you tell me my name without guessing?"],
        "direct_chat": ["How are you today?", "Are you hungry?", "Are you sleepy?", "What are you doing?", "Would you like to be friends?", "Please give me a hug.", "I am sad.", "I am scared.", "I am bored.", "I am tired.", "I feel alone.", "I need a friend.", "Stay with me for a moment.", "That was kind of you.", "You are funny!", "I like your wings.", "Your scales are beautiful.", "Can I pet your head?", "Do you get lonely?", "What makes you happy?", "What makes you worried?", "Can dragons cry?", "Are you feeling well?", "Let us take a nap.", "I want a story.", "It is sunny today.", "Snow is falling!", "The wind is strong.", "Look, a butterfly!", "I found a flower.", "There is a bird nearby.", "I can hear thunder.", "The moon is bright.", "Let's climb a hill.", "Let's sit under a tree.", "Shall we go swimming?", "What should we do next?", "Tell me one thing you love.", "Do you like music?", "Could you sing with me?"],
        "facts_variation": ["How many wings do most birds have?", "How many wheels does a bicycle have?", "What is one plus one?", "What is three plus one?", "Is fire hot or cold?", "Can a stone drink water?", "What color is a ripe banana?", "What do dogs say?", "What do cats say?", "What season comes after winter?", "Does ice melt when it gets warm?", "What is the opposite of big?", "What is the opposite of happy?", "What do we hear with?", "Why do we sleep?", "How can I help a friend feel better?"],
    }
    for family, prompts in groups.items():
        for prompt in prompts:
            add(family, [prompt])
    for name in ["Quinn", "Rosa", "Felix", "Tessa"]:
        add("validation_names", [f"Hello, I go by {name}.", "Repeat the name I introduced myself with."], [name.lower()], "validation")
    for prompt in ["Where is your cozy sleeping spot?", "What is your dragon name?", "Could you comfort a worried friend?", "Is your body covered in scales?", "Have I mentioned my address?", "What is one plus two?", "What snack could we share?", "You look like a rabbit to me."]:
        add("validation_variation", [prompt], split="validation")
    heldout = {p for _, prompts, _ in CASES for p in prompts}
    assert not any(p in heldout for row in rows for p in row["prompts"])
    assert len({r["id"] for r in rows}) == len(rows)
    random.Random(20261018).shuffle(rows)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("build/dragon/data"))
    p.add_argument("--url", default="http://127.0.0.1:11434")
    p.add_argument("--model", default="qwen3.5:9b")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--variant", choices=["original", "concise"], default="original")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = scenarios() if args.variant == "original" else concise_scenarios()
    persona = PERSONA
    predict = 96
    if args.variant == "concise":
        persona = (
            "You are Ember, a friendly little dragon. You live in a warm cave, eat apples, "
            "collect shiny pebbles, and enjoy exploring with friends. Reply in simple English, "
            "in ONE short sentence of 4 to 12 words. Answer the actual question directly. "
            "Do not add a follow-up question. Do not mention apples, pebbles or your cave unless relevant. "
            "Your name is always Ember. The user's name belongs to the user, not to you. "
            "When the user introduces themselves, acknowledge their exact name. Remember their most recent "
            "name and facts from this conversation. When asked about yourself, answer about the dragon. "
            "When asked about the user, use only what they told you; say you don't know otherwise. "
            "You collect stones but do not eat them. Stay in character. No stage directions."
        )
        predict = 64
    tags = request(args.url, "/api/tags")["models"]
    model = next(m for m in tags if m["name"] == args.model)
    metadata = {"teacher": model, "persona": persona, "seed": 42,
                "temperature": 0.7, "num_ctx": 2048, "num_predict": predict,
                "scenarios": rows}
    meta_path = args.output / "manifest.json"
    if meta_path.exists() and json.loads(meta_path.read_text()) != metadata:
        raise SystemExit("Dataset configuration changed; choose a new output directory")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")
    done = set()
    for split in ["train", "validation", "rejected"]:
        path = args.output / f"{split}.jsonl"
        if path.exists():
            done.update(json.loads(line)["id"] for line in path.read_text().splitlines())
    count = 0
    for row in rows:
        if row["id"] in done:
            continue
        messages = [{"role": "system", "content": persona}]
        reasons = []
        for prompt in row["prompts"]:
            messages.append({"role": "user", "content": prompt})
            raw = request(args.url, "/api/chat", {
                "model": args.model, "messages": messages, "stream": False,
                "think": False, "keep_alive": "30m",
                "options": {"temperature": 0.7, "seed": 42, "num_ctx": 2048, "num_predict": predict},
            })
            answer = raw["message"]["content"].strip()
            messages.append({"role": "assistant", "content": answer})
            if not answer or len(answer.split()) > 35 or raw.get("done_reason") != "stop":
                reasons.append("empty, long, or truncated answer")
            if not answer.isascii() or any(s in answer for s in ["<think>", "User:", "Assistant:"]):
                reasons.append("non-ASCII or role delimiter")
        if not all(word in answer.lower() for word in row["expected"]):
            reasons.append("missing expected memory fact")
        record = {**row, "messages": messages[1:], "rejection_reasons": reasons}
        target = "rejected" if reasons else row["split"]
        with (args.output / f"{target}.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"{len(done)+1}/{len(rows)} {target} {row['family']}: {answer}", flush=True)
        done.add(row["id"])
        count += 1
        if args.limit and count >= args.limit:
            break


if __name__ == "__main__":
    main()
