#!/usr/bin/env python3
"""Smoke-test the shared console against the actual shipped model on macOS/Linux."""
import os
from pathlib import Path
import re
import subprocess
import tempfile
from prepare_esp32ai import prepare

ROOT = Path(__file__).resolve().parents[1]


def main():
    with tempfile.TemporaryDirectory(prefix="tinyml-check-") as temporary:
        binary = Path(temporary) / "chat"
        prepare(Path(temporary))
        subprocess.run([
            os.environ.get("CXX", "c++"), "-std=c++17", "-O3",
            "-I", str(ROOT / "components/esp32ai/upstream"),
            "-I", temporary,
            str(ROOT / "main/console.cpp"), "-o", str(binary),
        ], check=True)
        messages = ["/temp 0", "Once upon a time", "There was a dragon named Ember.",
                    "/new", "Once upon a time", "/temp nan", "/temp 3", "/unknown",
                    "Привет", "x" * 257, "~!" * 100, "<|endoftext|>", "/stats", "/memory"]
        result = subprocess.run([
            str(binary), str(ROOT / "components/esp32ai/assets/model.bin"),
        ], input="\r\n".join(messages) + "\r\n", text=True,
            capture_output=True, check=True, timeout=60)
        output = result.stdout
        replies = re.findall(r"Story:(.*?)\n\[done\]", output, re.S)
        assert len(replies) == 3, output
        assert replies[0] == replies[2] and replies[0].strip(), output
        assert "history=0" in output, output
        assert output.count("[error]") == 7, output
        assert "Prompt exceeds" in output, output
        assert "[READY]" in output and "fp=a9bdd778" in output, output
        assert "[memory-begin]" in output and "[memory-end]" in output, output
        assert "HOST: allocation counters only" in output, output
        print("PASS: generation, independent prompts, reset, CRLF, invalid/oversized input")
        print("Greedy continuation:", replies[0].strip())
        bench = subprocess.run([str(binary), str(ROOT / "components/esp32ai/assets/model.bin")],
                               input="/temp 1\n/bench\n/bench oled\n/stats\n", text=True,
                               capture_output=True, check=True, timeout=60).stdout
        assert bench.count("[done] 200 tokens") == 2, bench
        assert bench.count("forwards=200") == 2, bench
        assert "temperature=1.00" in bench, bench
        assert "oled=0" in bench and "oled=1" in bench, bench
        bench_replies = re.findall(r"Story:(.*?)\n\[done\]", bench, re.S)
        assert bench_replies[0] == bench_replies[1], bench
        print("PASS: fixed 200-token benchmark, profile counts, greedy mode and temperature restore")
        # A stale/corrupt model must be rejected before the pointer-only loader.
        corrupt = Path(temporary) / "bad-model.bin"
        data = bytearray((ROOT / "components/esp32ai/assets/model.bin").read_bytes())
        data[64] ^= 1
        corrupt.write_bytes(data)
        bad = subprocess.run([str(binary), str(corrupt)], capture_output=True, text=True)
        assert bad.returncode == 1 and "fingerprint mismatch" in bad.stdout, bad.stdout
        corrupt.write_bytes(data[:128])
        bad = subprocess.run([str(binary), str(corrupt)], capture_output=True, text=True)
        assert bad.returncode == 1 and "too small" in bad.stdout, bad.stdout
        print("PASS: corrupt/truncated model rejected")
        staging = Path(temporary) / "staging"
        subprocess.run([os.environ.get("CC", "cc"), "-O3", "-DLLM_INT8_ACT=1",
                        str(ROOT / "components/esp32ai/upstream/host_verify/staging_verify.c"),
                        "-o", str(staging), "-lm"], check=True)
        staged = subprocess.run([str(staging), str(ROOT / "components/esp32ai/assets/model.bin")],
                                capture_output=True, text=True, check=True)
        assert "PASS (0 failures)" in staged.stdout, staged.stdout
        print("PASS: upstream int8 staging, alignment, matvec hooks and model header gates")

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
