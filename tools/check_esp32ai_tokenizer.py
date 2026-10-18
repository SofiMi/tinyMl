#!/usr/bin/env python3
"""Compare the firmware TinyStories encoder with Hugging Face tokenizers."""
import os
from pathlib import Path
import random
import subprocess
import tempfile

from tokenizers import Tokenizer
from prepare_esp32ai import ROOT, prepare


def main():
    tok = Tokenizer.from_file(str(ROOT / 'components/esp32ai/assets/tokenizer.json'))
    probes = ['', 'Once upon a time', 'There was a dragon named Ember.',
              "I'm fine", "DON'T", "we'll see", '  hello  world   ',
              'a\tb', 'line\nbreak', '18 g, sour.', '  leading and trailing  ']
    probes += [chr(i) for i in range(32, 127)]
    rng = random.Random(42)
    alphabet = ''.join(chr(i) for i in range(32, 127)) + '\n\t\r'
    probes += [''.join(rng.choices(alphabet, k=rng.randrange(1, 257))) for _ in range(1000)]
    with tempfile.TemporaryDirectory(prefix='tinyml-tokenizer-') as temp:
        work = Path(temp)
        prepare(work)
        binary = work / 'conformance'
        subprocess.run([os.environ.get('CC', 'cc'), '-O2', '-std=c11', '-Wall',
                        '-Wextra', '-Werror', '-o', str(binary),
                        str(ROOT / 'components/esp32ai/upstream/host_verify/tokenizer_conformance.c')],
                       check=True)
        for start in range(0, len(probes), 100):
            batch = probes[start:start+100]
            result = subprocess.run([str(binary), str(work / 'tokenizer.btk'), *batch],
                                    text=True, capture_output=True, check=True)
            lines = result.stdout.split('\n')[:-1]
            assert len(lines) == len(batch)
            for text, line in zip(batch, lines):
                actual = [int(x) for x in line.split()]
                expected = tok.encode(text).ids
                assert actual == expected, (repr(text), actual, expected)
        result = subprocess.run([str(binary), str(work / 'tokenizer.btk'), 'café', 'x' * 513],
                                text=True, capture_output=True, check=True)
        assert result.stdout.splitlines() == ['ERROR:-3', 'ERROR:-2'], result.stdout
    print(f'PASS: {len(probes)} ASCII inputs match Hugging Face; unsupported inputs rejected')


if __name__ == '__main__':
    main()
