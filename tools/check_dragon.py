#!/usr/bin/env python3
"""Exercise the actual shared C++ character and persistence via its console."""
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="tinyml-dragon-") as temporary:
        work = Path(temporary)
        cxx = [os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-Wall", "-Wextra"]
        state = work / "state-test"
        subprocess.run(cxx + ["-I", str(ROOT / "main"),
                              str(ROOT / "tests/dragon/state_test.cpp"),
                              str(ROOT / "main/dragon.cpp"), "-o", str(state)], check=True)
        subprocess.run([str(state)], check=True)
        binary = work / "chat"
        subprocess.run(cxx + ["-pthread", "-I", str(ROOT / "tools/host"),
                              "-I", str(ROOT / "components/tinytalk/include"),
                              *[str(ROOT / "main" / f) for f in ("console.cpp", "dragon.cpp", "dragon_store.cpp")],
                              str(ROOT / "components/tinytalk/llm.cpp"), "-o", str(binary)], check=True)
        memory = work / "memory.bin"

        def run(messages, path=memory):
            result = subprocess.run([
                str(binary), str(ROOT / "components/tinytalk/assets/model_neo_q4.bin"),
                str(ROOT / "components/tinytalk/assets/tok_neo.bin"),
            ], input="\r\n".join(messages + ["/quit"]) + "\r\n", text=True,
                capture_output=True, check=True, timeout=60,
                env={**os.environ, "TINYML_DRAGON_MEMORY": str(path)})
            return result.stdout

        original = json.loads((ROOT / "tests/dragon/sofia-loop-v1.json").read_text())
        output = run(original)
        replies = re.findall(r"^Ember: (.*)$", output, re.M)
        assert len(replies) == len(original), output
        assert "I am Ember" in replies[0] and "I am Ember" in replies[1], output
        assert "Hello, Sofia" in replies[2] and "Hello, Sofia" in replies[3], output
        assert "You are Sofia" in replies[4] and "= 2." in replies[5], output
        assert len(set(replies[6:])) == 3, output
        assert "[model]" not in output, output  # no raw model allocation in default mode
        print("PASS: original nine-turn Sofia transcript, no identity swap or fixed reply loop")
        for prompt, reply in zip(original, replies):
            print(f"  {prompt} -> {reply}")

        output = run(["What is my name?", "I like green", "/new", "What is my favorite color?", "/memory"])
        assert "Your name is Sofia" in output and "You like green" in output, output
        assert "Your favorite color: green" in output, output
        assert memory.stat().st_size == 204
        output = run(["What is my name?", "What is my favorite color?", "/mode raw", "/temp 0",
                      "I am Alex", "/mode dragon", "What is my name?", "/forget", "What is my name?"])
        assert "Bot:" in output and output.count("Your name is Sofia") == 2, output
        assert "don't know your name" in output, output
        output = run(["What is my name?", "What is my favorite color?"])
        assert "don't know your name" in output and "don't know your favorite color" in output, output
        print("PASS: saved facts across processes, /new retains, /forget erases, raw mode isolation")

        memory.write_bytes(b"broken")
        output = run(["My name is Alice", "What is my name?"])
        assert "invalid saved facts" in output and "Your name is Alice" in output, output
        assert memory.read_bytes() == b"broken"  # no automatic destructive recovery
        run(["/forget", "My name is Sofia"])
        assert "Your name is Sofia" in run(["What is my name?"])
        output = run(["My name is Alice", "What is my name?"], work / "missing" / "memory.bin")
        assert "[warning]" in output and "Your name is Alice" in output, output
        output = run(["Привет", "x" * 257, "/temp 1", "/unknown"])
        assert output.count("[error]") == 4, output
        print("PASS: corrupt memory recovery, failed save stays usable, invalid input")


if __name__ == "__main__":
    main()
