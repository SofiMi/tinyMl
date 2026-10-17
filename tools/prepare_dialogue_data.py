#!/usr/bin/env python3
"""Build a pinned short-chat subset and controlled instruction-following data."""
import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

REVISION = "f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc"
NEUTRAL = "You are a helpful assistant. Answer in short, simple English."
NAMES = "Ada Ben Cora Dan Ella Finn Gail Hugo Iris Jack Kara Leo Maya Noah Opal Paul Rose Sam Tess Will Amy Bob Cody Dora Evan Fred Gina Hank Ivan June Kate Luis Mina Nate Owen Pete Rita Seth Tina Wade".split()
ANIMALS = "cat rabbit fox mouse bird turtle".split()
FOODS = "berries carrots bread rice peaches nuts".split()
HOMES = "garden forest house meadow hill island".split()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def conversation(identifier, family, system, pairs, **extra):
    messages = [{"role": "system", "content": system}]
    for q, a in pairs:
        messages.extend([{"role": "user", "content": q}, {"role": "assistant", "content": a}])
    return dict(id=identifier, family=family, messages=messages, **extra)


def controlled(seed, count, names, split):
    rng = random.Random(seed)
    rows = []
    for i in range(count):
        bot, user, friend = rng.sample(names, 3)
        animal, food, home = rng.choice(ANIMALS), rng.choice(FOODS), rng.choice(HOMES)
        system = f"You are {bot}, a friendly {animal}. You live in a {home} and enjoy {food}. Answer in short, simple English. Do not invent facts about the user."
        concise = rng.choice([True, False])
        suffix = " Reply with the name only." if concise else ""
        kind = i % 10
        if kind == 0:
            family = "assistant_name"
            pairs = [(rng.choice(["What is your name?", "Who are you called?"])+suffix, bot if concise else f"My name is {bot}.")]
        elif kind == 1:
            family = "user_name"
            pairs = [(f"My name is {user}.", f"Hello, {user}!"), ("What is my name?"+suffix, user if concise else f"Your name is {user}.")]
        elif kind == 2:
            family = "two_names"
            pairs = [(f"Hello, I am {user}.", f"Hi, {user}! I am {bot}."), ("What are our names?", f"I am {bot}. You are {user}.")]
        elif kind == 3:
            family = "correction"
            pairs = [(f"My name is {friend}.", f"Hello, {friend}!"), (f"Actually, my name is {user}.", f"Thanks for correcting me, {user}."), ("What is my name now?"+suffix, user if concise else f"Your name is {user}.")]
        elif kind == 4:
            family = "friend"
            pairs = [(f"I am {user}. My friend is {friend}.", f"Hello, {user}!"), ("What is my friend's name?"+suffix, friend if concise else f"Your friend is {friend}.")]
        elif kind == 5:
            family = "species"
            pairs = [("What animal are you?"+(" Reply with one word." if concise else ""), animal if concise else f"I am a friendly {animal}.")]
        elif kind == 6:
            family = "system_fact"
            pairs = [("What food do you enjoy?"+(" Reply with one word." if concise else ""), food if concise else f"I enjoy {food}.")]
        elif kind == 7:
            family = "unknown"
            pairs = [(rng.choice(["What is my name?", "Where do I live?", "What is my favorite food?"]), "I don't know yet. Can you tell me?")]
        elif kind == 8:
            family = "ownership"
            a, b = rng.sample(["book", "ball", "cup", "hat", "key", "pencil"], 2)
            pairs = [(f"The {a} is mine. The {b} is yours.", f"You have the {a}. I have the {b}."), ("Which object is mine?"+(" Reply with one word." if concise else ""), a if concise else f"The {a} is yours.")]
        else:
            family = "negation"
            a, b = rng.sample(["red", "blue", "yellow", "purple", "white", "black"], 2)
            pairs = [(f"I like {a}, not {b}.", f"You like {a}."), ("What color do I like?"+(" Reply with one word." if concise else ""), a if concise else f"You like {a}.")]
        rows.append(conversation(f"controlled-{split}-{i}", family, system, pairs, source="deterministic-ground-truth-v1"))
    return rows


def evaluation():
    cases = []
    for i, (bot, user, other) in enumerate([("Ember", "Sofia", "Mark"), ("Flint", "Alice", "James"), ("Nimbus", "Olivia", "Henry"), ("Spark", "Chloe", "Oscar")]):
        system = f"You are {bot}, a friendly dragon. You live in a cave and enjoy apples. Answer in short, simple English. Do not invent facts about the user."
        tasks = [
            ("assistant_name", ["Tell me your name. Respond with the name alone."], [bot]),
            ("species", ["What kind of creature are you? Give only the animal name."], ["dragon"]),
            ("food", ["Name your favorite food. One word only."], ["apples"]),
            ("user_name", [f"You can call me {user}.", "Tell me my name. Respond with the name alone."], [user]),
            ("correction", [f"I am {other}.", f"Sorry, that was wrong. I am {user}.", "Give my corrected name and nothing else."], [user]),
            ("friend", [f"You can call me {user}. My friend is {other}.", "Name my friend. Give only their name."], [other]),
            ("ownership", ["The pear belongs to me. The apple belongs to you.", "Which fruit belongs to me? One word only."], ["pear"]),
            ("negation", ["My favorite color is green. It is not orange.", "Give my favorite color in one word."], ["green"]),
            ("arithmetic", ["What is two plus three? Give only the number."], ["5", "five"]),
            ("natural", [f"Hello! My name is {user}.", "Tell me about yourself."], []),
        ]
        for family, prompts, answers in tasks:
            cases.append(dict(id=f"{family}-{i}", family=family, system=system, prompts=prompts, answers=answers))
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shard", type=Path, default=Path.home()/".cache/tinyml-smol-smoltalk/data/train-00000-of-00004.parquet")
    p.add_argument("--output", type=Path, default=Path("training/dialogue-v1"))
    args = p.parse_args()
    generic = {"train": [], "validation": []}
    seen = set()
    for index, row in enumerate(pq.read_table(args.shard).to_pylist()):
        messages = row["messages"]
        if not messages or messages[0]["role"] != "user":
            continue
        key = digest(" ".join(messages[0]["content"].lower().split()))
        if key in seen:
            continue
        pairs = []
        for j in range(0, min(len(messages)-1, 6), 2):
            q, a = messages[j:j+2]
            if q["role"] != "user" or a["role"] != "assistant":
                break
            if any(not 2 <= len(m["content"].split()) <= 40 or "```" in m["content"] for m in [q, a]):
                break
            pairs.append((q["content"], a["content"]))
        if not pairs:
            continue
        seen.add(key)
        split = "validation" if int(key[:8], 16) % 10 == 0 else "train"
        generic[split].append(conversation(f"smol-{key}", "general", NEUTRAL, pairs, source=row["source"], shard_row=index))
    rng = random.Random(42)
    for split, limit in [("train", 1400), ("validation", 160)]:
        rng.shuffle(generic[split])
        generic[split] = generic[split][:limit]
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    metadata = {"source": "https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk", "revision": REVISION,
                "shard": args.shard.name, "shard_sha256": hashlib.sha256(args.shard.read_bytes()).hexdigest(),
                "upstream_declared_license": "apache-2.0", "seed": 42,
                "split": "General: normalized first-user SHA bucket; controlled templates: disjoint names across splits and evaluation, no dragon in controlled training. No claim of absence from general data or pretraining.",
                "teacher": "Public synthetic dialogue subset plus deterministic factual templates; no new teacher inference or logit distillation.",
                "splits": {}}
    for split, names, n, seed in [("train", NAMES[:32], 1600, 42), ("validation", NAMES[32:], 200, 43)]:
        rows = generic[split]+controlled(seed, n, names, split)
        rng.shuffle(rows)
        content = "".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows)
        (output/f"{split}.jsonl").write_text(content)
        metadata["splits"][split] = {"rows": len(rows), "families": dict(Counter(r["family"] for r in rows)), "sha256": digest(content)}
    (output/"provenance.json").write_text(json.dumps(metadata, indent=2)+"\n")
    cases = Path("tests/dragon/dialogue-v1.json")
    cases.write_text(json.dumps(evaluation(), indent=2)+"\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
