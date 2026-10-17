#!/usr/bin/env python3
"""Compare shipped Q4 inference with its local FP32 source checkpoint.

Install torch, transformers, safetensors and numpy in a separate environment.
Pass --upstream pointing to cardputer-ai at the revision in UPSTREAM.md.
No network/model downloads are performed. Results go to build/evaluation.
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import argparse, hashlib, importlib.util, json, re, struct, subprocess
from pathlib import Path
import torch
import transformers
from transformers import GPTNeoForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--upstream', type=Path, required=True)
parser.add_argument('--checkpoint', type=Path)
parser.add_argument('--output', type=Path, default=ROOT / 'build/evaluation')
args = parser.parse_args()
UP = args.upstream.resolve()
OUT = args.output.resolve()
OUT.mkdir(parents=True, exist_ok=True)
CHECKPOINT = args.checkpoint or UP / 'data/chat_model_8m'
subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-pthread',
    '-I', str(ROOT/'tools/host'), '-I', str(ROOT/'components/tinytalk/include'),
    str(UP/'tools/host/host_test.cpp'), str(ROOT/'components/tinytalk/llm.cpp'),
    '-o', str(OUT/'upstream-host')], check=True)
torch.set_num_threads(2)
torch.manual_seed(0)
spec = importlib.util.spec_from_file_location('converter', UP / 'tools/convert_tinystories_instruct.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)

blob = (ROOT / 'components/tinytalk/assets/tok_neo.bin').read_bytes()
magic, size, maxlen, eos, merges = struct.unpack_from('<Iiiii', blob)
offset = 20 + 512 + merges * 6
pieces = []
for _ in range(size):
    length, = struct.unpack_from('<i', blob, offset)
    offset += 4
    pieces.append(blob[offset:offset+length])
    offset += length
assert offset == len(blob)
vocab = json.loads((CHECKPOINT / 'vocab.json').read_text())
u2b = {v:k for k,v in converter.bytes_to_unicode().items()}
byte_to_id = {converter.piece_to_bytes(s, u2b):i for s,i in vocab.items() if i != 50256}
kept = [50256 if i == eos else byte_to_id[p] for i,p in enumerate(pieces)]
assert kept == sorted(kept)
converted = OUT / 'reconverted.bin'
converter.convert_model(CHECKPOINT, converted, kept, 256)
same = converted.read_bytes() == (ROOT / 'components/tinytalk/assets/model_neo_q4.bin').read_bytes()
print('CHECKPOINT_RECONVERSION_IDENTICAL:', same, flush=True)
assert same, 'Cannot attribute comparison to the shipped checkpoint'

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True)
model = GPTNeoForCausalLM.from_pretrained(CHECKPOINT, local_files_only=True, attn_implementation='eager').eval()

def hf_encode(prompt):
    result = []
    for i, segment in enumerate(prompt.split('<|eos|>')):
        if i: result.append(50256)
        result.extend(tokenizer.encode(segment, add_special_tokens=False))
    return result

def hf_run(ids):
    tokens = torch.tensor([ids])
    generated = []
    with torch.inference_mode():
        first_logits = model(tokens, use_cache=False).logits[0,-1]
        full_top = first_logits.topk(10).indices.tolist()
        kept_top = [kept[i] for i in first_logits[kept].topk(10).indices.tolist()]
        for _ in range(48):
            logits = model(tokens, use_cache=False).logits[0,-1]
            nxt = logits.argmax().item()
            if nxt == 50256: break
            generated.append(nxt)
            tokens = torch.cat([tokens, torch.tensor([[nxt]])], dim=1)
    return tokenizer.decode(generated), full_top, kept_top

def q4_run(prompt, kv=72):
    result = subprocess.run([str(OUT / 'upstream-host'),
        str(ROOT / 'components/tinytalk/assets/model_neo_q4.bin'),
        str(ROOT / 'components/tinytalk/assets/tok_neo.bin'), prompt,
        '--max','48','--temp','0','--kv',str(kv),'--slide','--top10'],
        text=True, capture_output=True, check=True, timeout=60)
    ids = list(map(int, re.search(r'prompt tokens \(\d+\):(.*)', result.stderr)[1].split()))
    first = re.search(r'pos \d+ top10:(.*)', result.stderr)[1]
    top = [kept[int(v.split(':')[0])] for v in first.split()]
    return result.stdout.rstrip('\n'), [kept[i] for i in ids], top, result.stderr

cases = [
    ('dragon_intro', 'User: You are Ember, a friendly little dragon. Say hello.\nBot:'),
    ('dragon_name', 'User: Your name is Ember. You are a dragon. What is your name?\nBot:'),
    ('dragon_memory', 'User: You are Ember, a friendly dragon.\nBot: Hello! My name is Ember.<|eos|>\nUser: what is your name?\nBot:'),
    ('hello', 'User: hello\nBot:'),
    ('name_alone', 'User: my name is sofia\nBot:'),
    ('language', 'User: what language do you speak?\nBot:'),
    ('name_memory_clean', 'User: my name is sofia\nBot: Nice to meet you, Sofia.<|eos|>\nUser: what is my name?\nBot:'),
    ('user_recorded_second', "User: hello\nBot: It's me, your name is Mark.<|eos|>\nUser: my name is sofia\nBot:"),
    ('user_recorded_last', "User: my name is sofia\nBot: My name is Lily. It's nice to meet you.<|eos|>\nUser: what language do you speak?\nBot: I'm a language, my language is called language.<|eos|>\nUser: what is my name?\nBot:"),
]
for question in ['what color is the sky?', 'what color is grass?', 'what sound does a dog make?',
                 'what sound does a cat make?', 'how many legs does a spider have?',
                 'what is a baby dog called?', 'what is the opposite of hot?', 'what do bees make?']:
    cases.append((question, 'User: ' + question + '\nBot:'))

report = {'checkpoint_reconversion_identical': same, 'torch': torch.__version__,
          'transformers': transformers.__version__,
          'upstream_revision': subprocess.check_output(['git','rev-parse','HEAD'], cwd=UP, text=True).strip(),
          'weights_sha256': hashlib.sha256(converted.read_bytes()).hexdigest(),
          'scope': 'CPU, greedy, same fixed prompts; FP32 full vocabulary vs shipped Q4 weights/int8 activations/int4 KV', 'cases': []}
for name,prompt in cases:
    q4, qids, qtop, trace = q4_run(prompt)
    hfids = hf_encode(prompt)
    # Compare the same input token IDs. Pruning legitimately splits some words
    # differently, so keep a separate full-tokenizer baseline when that happens.
    assert tokenizer.decode(qids) == prompt.replace('<|eos|>', '<|endoftext|>')
    fp32, ftop, ktop = hf_run(qids)
    native_fp32 = fp32 if qids == hfids else hf_run(hfids)[0]
    q256, q256ids, _, _ = q4_run(prompt, 256)
    item = dict(name=name, prompt=prompt, tokens=len(qids), tokenizer_match=qids==hfids,
                tokenizer_roundtrip=True, fp32_native_tokenization=native_fp32,
                fp32=fp32, q4=q4, q4_kv256=q256, q4_first_top10=qtop,
                fp32_first_top10=ftop, fp32_kept_first_top10=ktop)
    report['cases'].append(item)
    print(json.dumps(item, ensure_ascii=False), flush=True)
    (OUT/'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
# Compare the shared console's history formatting against the upstream harness.
subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-pthread',
    '-I', str(ROOT/'tools/host'), '-I', str(ROOT/'components/tinytalk/include'),
    str(ROOT/'main/console.cpp'), str(ROOT/'components/tinytalk/llm.cpp'),
    '-o', str(OUT/'console')], check=True)
questions = ['hello', 'my name is sofia', 'what language do you speak?', 'what is my name?']
console = subprocess.run([str(OUT/'console'), str(ROOT/'components/tinytalk/assets/model_neo_q4.bin'),
    str(ROOT/'components/tinytalk/assets/tok_neo.bin')],
    input='/temp 0\n' + '\n'.join(questions) + '\n', text=True,
    capture_output=True, check=True, timeout=60)
(OUT/'console-transcript.txt').write_text(console.stdout)
replies = re.findall(r'Bot:(.*?)\n\[done\]', console.stdout, re.S)
assert len(replies) == len(questions), console.stdout
history = []
report['console_parity'] = []
for question, reply in zip(questions, replies):
    prompt = 'User: ' + question + '\nBot:'
    for user,bot in reversed(history):
        candidate = 'User: ' + user + '\nBot:' + bot + '<|eos|>\n' + prompt
        if len(hf_encode(candidate)) > 64: break
        prompt = candidate
    reference, _, _, _ = q4_run(prompt)
    match = reply == reference
    report['console_parity'].append(dict(question=question, prompt=prompt,
                                        console=reply, reference=reference, match=match))
    assert match, report['console_parity'][-1]
    history.append((question,reply))
(OUT/'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
print('PASS: console/history matches upstream on all four turns', flush=True)
print('FINISHED', flush=True)
