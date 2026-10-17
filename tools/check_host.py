#!/usr/bin/env python3
"""Smoke-test the shared console against the actual shipped model on macOS/Linux."""
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="tinyml-check-") as temporary:
        binary = Path(temporary) / "chat"
        subprocess.run([
            os.environ.get("CXX", "c++"), "-std=c++17", "-O2", "-pthread",
            "-I", str(ROOT / "tools/host"),
            "-I", str(ROOT / "components/tinytalk/include"),
            str(ROOT / "main/console.cpp"),
            str(ROOT / "components/tinytalk/llm.cpp"), "-o", str(binary),
        ], check=True)
        messages = ["/temp 0", "Hi!", "How are you?", "/stats", "/new", "Hi!",
                    "/temp nan", "/temp 3", "/unknown", "Привет", "x" * 257,
                    "zxq " * 60, "/stats"]
        result = subprocess.run([
            str(binary), str(ROOT / "components/tinytalk/assets/model_neo_q4.bin"),
            str(ROOT / "components/tinytalk/assets/tok_neo.bin"),
        ], input="\r\n".join(messages) + "\r\n", text=True,
            capture_output=True, check=True, timeout=60)
        output = result.stdout
        replies = re.findall(r"Bot:(.*?)\n\[done\]", output, re.S)
        assert len(replies) == 3, output
        assert replies[0] == replies[2] and replies[0].strip(), output
        assert "history=2" in output and "history=1" in output, output
        assert output.count("[error]") == 6, output
        assert "Question exceeds" in output, output
        assert "[READY]" in output, output
        print("PASS: model generation, multi-turn history, reset, CRLF, invalid/oversized input")
        print("Greedy Hi! reply:", replies[0].strip())

        test_source = Path(temporary) / "display_test.cpp"
        test_source.write_text('''
#include "text_window.h"
#include <cassert>
#include <cstring>
int main() {
    TextWindow w;
    w.append("123456789012345678901X");
    assert(std::strcmp(w.line(0), "123456789012345678901") == 0);
    assert(std::strcmp(w.line(1), "X") == 0);
    w.clear();
    w.append("a\\nb\\nc\\nd\\ne\\nf\\ng\\nh");
    assert(std::strcmp(w.line(0), "b") == 0);
    assert(std::strcmp(w.line(6), "h") == 0);
    w.clear();
    for (int i=0; i<7; ++i) assert(w.line(i)[0] == 0);
}
''')
        test_binary = Path(temporary) / "display_test"
        subprocess.run([os.environ.get("CXX", "c++"), "-std=c++17", "-I",
                        str(ROOT / "main"), str(test_source), "-o", str(test_binary)], check=True)
        subprocess.run([str(test_binary)], check=True)
        print("PASS: OLED text wrapping, scrolling and clearing")


if __name__ == "__main__":
    main()
