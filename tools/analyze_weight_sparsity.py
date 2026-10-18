#!/usr/bin/env python3
"""Read-only sparsity audit of local model weights, with ideal bitmap estimates."""
import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import struct

import numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize

ROOT = Path(__file__).resolve().parents[1]
LOCK = json.loads((ROOT/'tools/small_models.lock.json').read_text())
RULES = [('exact_zero', 0., False), ('abs_0.0001', .0001, False),
         ('abs_0.001', .001, False), ('abs_0.01', .01, False),
         ('rms_0.01', .01, True), ('rms_0.05', .05, True),
         ('rms_0.1', .1, True), ('rms_0.25', .25, True), ('rms_0.5', .5, True)]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while data := stream.read(2**20):
            h.update(data)
    return h.hexdigest()


def tensor(name, values, kind, bits, stored):
    return name, np.asarray(values, dtype=np.float32), kind, bits, stored


def gguf_tensors(path):
    reader = GGUFReader(path)
    bits = {'F32': 32, 'F16': 16, 'Q4_K': 4, 'Q6_K': 6, 'Q8_0': 8, 'Q4_0': 4, 'Q5_0': 5}
    for t in reader.tensors:
        kind = t.tensor_type.name
        values = dequantize(t.data, t.tensor_type)
        assert values.size == t.n_elements
        yield tensor(t.name, values, kind, bits[kind], t.n_bytes)


def torch_tensors(path, model, exclusions):
    import torch
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint['model_state'] if model == 'vagh' else checkpoint
    seen = set()
    for name, value in state.items():
        if model == 'gpt2' and (name.endswith('.attn.bias') or name.endswith('.attn.masked_bias')):
            exclusions.append(name + ': non-learned causal attention buffer')
            continue
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            exclusions.append(name + ': non-floating or non-tensor')
            continue
        identity = (value.untyped_storage().data_ptr(), value.storage_offset(),
                    tuple(value.shape), tuple(value.stride()))
        if identity in seen:
            exclusions.append(name + ': tied/shared tensor already counted')
            continue
        seen.add(identity)
        assert value.dtype == torch.float32
        yield tensor(name, value.numpy(), 'F32', 32, value.numel()*4)


def ple_tensors(path):
    data = path.read_bytes()
    magic, version, pos, flags = struct.unpack_from('<4I', data)
    assert magic == 0x00454c50 and version == 1 and flags == 1
    vocab, out_vocab = struct.unpack_from('<2I', data, 16)
    dim, layers, heads, ffn, ple, seq, group = struct.unpack_from('<7i', data, 24)
    assert out_vocab <= vocab

    def f(name, size):
        nonlocal pos
        values = np.frombuffer(data, '<f4', count=size, offset=pos)
        pos += size*4
        return tensor(name, values, 'F32', 32, size*4)

    def q(name, rows, cols):
        nonlocal pos
        start = pos
        group = struct.unpack_from('<i', data, pos)[0]
        assert group > 0
        pos += 4
        row_bytes, groups = (cols+1)//2, (cols+group-1)//group
        packed = np.frombuffer(data, np.uint8, count=rows*row_bytes, offset=pos).reshape(rows,row_bytes)
        pos += rows*row_bytes
        scales = np.frombuffer(data, '<f2', count=rows*groups, offset=pos).reshape(rows,groups).astype(np.float32)
        pos += rows*groups*2
        codes = np.empty((rows,row_bytes*2), dtype=np.int8)
        codes[:,0::2] = (packed & 15).astype(np.int8)-8
        codes[:,1::2] = (packed >> 4).astype(np.int8)-8
        values = codes[:,:cols].astype(np.float32)*np.repeat(scales,group,axis=1)[:,:cols]
        return tensor(name,values,'PLE_Q4',4,pos-start)

    yield q('tok_emb',vocab,dim)
    yield q('ple_model_proj',layers*ple,dim)
    yield f('ple_proj_norm',ple)
    yield q('ple_table',vocab,layers*ple)
    for layer in range(layers):
        prefix = f'blk.{layer}.'
        yield f(prefix+'attn_norm',dim)
        yield q(prefix+'qkv',3*dim,dim)
        yield q(prefix+'attn_proj',dim,dim)
        yield f(prefix+'ffn_norm',dim)
        for name,rows,cols in [('gate',ffn,dim),('up',ffn,dim),('down',dim,ffn),
                               ('ple_gate',ple,dim),('ple_proj',dim,ple)]:
            yield q(prefix+name,rows,cols)
        yield f(prefix+'ple_norm',dim)
    yield f('out_norm',dim)
    assert pos == len(data), (pos,len(data))


def tinytalk_tensors(path):
    data = path.read_bytes()
    assert data[:4] == b'CRDP' and struct.unpack_from('<I',data,4)[0] == 3
    dim, hidden, layers, heads, kv_heads, vocab, seq = struct.unpack_from('<7i',data,8)
    assert tuple(data[36:39]) == (1,4,2)
    pos = 64

    def f(name,shape):
        nonlocal pos
        count = math.prod(shape)
        values = np.frombuffer(data,'<f4',count=count,offset=pos).reshape(shape)
        pos += count*4
        return tensor(name,values,'F32',32,count*4)

    for name,shape in [('ln1_g',(layers,dim)),('ln1_b',(layers,dim)),
                       ('ln2_g',(layers,dim)),('ln2_b',(layers,dim)),
                       ('lnf_g',(dim,)),('lnf_b',(dim,)),('out_bias',(layers,dim)),
                       ('fc_bias',(layers,hidden)),('proj_bias',(layers,dim)),('wpe',(seq,dim))]:
        yield f(name,shape)
    pos = (pos+15)//16*16

    def q(name,rows,cols):
        nonlocal pos
        assert cols % 32 == 0
        groups = cols//32
        nibble_offset = (groups*2+15)//16*16
        stride = nibble_offset+cols//2
        block = np.frombuffer(data,np.uint8,count=rows*stride,offset=pos).reshape(rows,stride)
        pos += rows*stride
        scales = (block[:,:groups*2].copy().view('<u2').astype(np.uint32)<<16).view(np.float32)
        packed = block[:,nibble_offset:].reshape(rows,groups,16)
        codes = np.concatenate([packed & 15,packed >> 4],axis=2).astype(np.int8)-8
        values = (codes.astype(np.float32)*scales[:,:,None]).reshape(rows,cols)
        return tensor(name,values,'TINY_Q4_BF16',4,rows*stride)

    yield q('wte',vocab,dim)
    for name,rows,cols in [('wq',dim,dim),('wk',dim,dim),('wv',dim,dim),
                           ('wo',dim,dim),('w_fc',hidden,dim),('w_proj',dim,hidden)]:
        for layer in range(layers):
            yield q(f'blk.{layer}.{name}',rows,cols)
    assert pos == len(data),(pos,len(data))


def audit(key, tensors):
    details = []
    totals = defaultdict(float)
    by_type = defaultdict(lambda: defaultdict(int))
    for name,values,kind,bits,stored in tensors:
        shape = list(values.shape)
        values = values.reshape(-1)
        assert np.isfinite(values).all(), (key,name,'non-finite weights')
        n = int(values.size)
        absolute = np.abs(values)
        energy = float(np.sum(np.square(values,dtype=np.float64)))
        rms = math.sqrt(energy/n)
        overhead = stored - (n*bits+7)//8
        assert overhead >= 0
        zeros = int(np.count_nonzero(values == 0))
        base = dict(model=key,tensor=name,shape=shape,storage=kind,code_bits=bits,
                    parameters=n,stored_bytes=stored,exact_zeros=zeros,rms=rms,
                    min_value=float(values.min()),max_value=float(values.max()),
                    mean_abs=float(absolute.mean(dtype=np.float64)))
        by_type[kind]['parameters'] += n
        by_type[kind]['exact_zeros'] += zeros
        by_type[kind]['bytes'] += stored
        totals['parameters'] += n
        totals['stored_bytes'] += stored
        totals['squared_weight_sum'] += energy
        for rule,threshold,relative in RULES:
            limit = threshold*rms if relative else threshold
            mask = absolute <= limit
            count = int(np.count_nonzero(mask))
            removed = float(np.sum(np.square(values[mask],dtype=np.float64)))
            # Ideal per-tensor stream, preserving existing scale/min metadata.
            # No offset tables, alignment or per-tensor format-selection flags.
            sparse = overhead + (n+7)//8 + ((n-count)*bits+7)//8
            base[rule+'_count'] = count
            base[rule+'_pct'] = count/n*100
            base[rule+'_sparse_bytes'] = sparse
            totals[rule+'_count'] += count
            totals[rule+'_removed_squared_sum'] += removed
            totals[rule+'_all_sparse_bytes'] += sparse
            totals[rule+'_selective_sparse_bytes'] += min(stored,sparse)
        details.append(base)
    result = dict(model=key, tensors=len(details),parameters=int(totals['parameters']),
                  stored_tensor_bytes=int(totals['stored_bytes']),by_type=dict(by_type),thresholds=[])
    for rule,threshold,relative in RULES:
        count = int(totals[rule+'_count'])
        result['thresholds'].append(dict(rule=rule,threshold=threshold,relative_to_tensor_rms=relative,
            count=count,pct=count/result['parameters']*100,
            removed_squared_weight_pct=totals[rule+'_removed_squared_sum']/totals['squared_weight_sum']*100,
            all_sparse_bytes=int(totals[rule+'_all_sparse_bytes']),
            selective_sparse_bytes=int(totals[rule+'_selective_sparse_bytes']),
            selective_saved_pct=(1-totals[rule+'_selective_sparse_bytes']/totals['stored_bytes'])*100))
    print(key,result['parameters'],'weights; exact zeros',result['thresholds'][0]['pct'],flush=True)
    return result, details


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    models,details=[],[]
    for key in ['supra','supra-q8','stentor','stentor-q8','stories','vagh','gpt2','ple','tinytalk']:
        excluded=[]
        if key in LOCK and key != 'vagh':
            spec=next(f for f in LOCK[key]['files'] if f['rfilename'].endswith('.gguf'))
            path=ROOT/'build/model-playground'/key/spec['rfilename']
            expected=spec['lfs']['sha256']
            tensors=gguf_tensors(path)
        elif key == 'vagh':
            path=ROOT/'build/model-playground/vagh/checkpoints/best_checkpoint.pt'
            expected=next(f['lfs']['sha256'] for f in LOCK[key]['files'] if f['rfilename'].endswith('.pt'))
            tensors=torch_tensors(path,key,excluded)
        elif key == 'gpt2':
            path=ROOT/'build/gpt2-small/model/pytorch_model.bin'
            expected='7c5d3f4b8b76583b422fcb9189ad6c89d5d97a094541ce8932dce3ecabde1421'
            tensors=torch_tensors(path,key,excluded)
        elif key == 'ple':
            path=ROOT/'components/esp32ai/assets/model.bin'
            expected='1d8326c05c383ccfa615f5455575802817cb453dbc7ab28875d41a9dbb45477e'
            tensors=ple_tensors(path)
        else:
            path=ROOT/'components/tinytalk/assets/model_neo_q4.bin'
            expected='183fe4c212fd0fd17a8e048ee3eab4a0095a039df290f0c58cafff70ca17fe2a'
            tensors=tinytalk_tensors(path)
        assert sha(path)==expected, path
        result,rows=audit(key,tensors)
        result.update(source=str(path.relative_to(ROOT)),sha256=expected,excluded=excluded)
        expected_count={'supra':100684032,'supra-q8':100684032,'stentor':30419712,
                        'stentor-q8':30419712,'stories':58073600,'vagh':42615808,
                        'gpt2':124439808,'ple':28869920,'tinytalk':9687808}.get(key)
        if expected_count is not None:
            assert result['parameters']==expected_count,(key,result['parameters'],expected_count)
        models.append(result)
        details.extend(rows)
        (a.output/f'{key}.json').write_text(json.dumps(result,indent=2)+'\n')
    metadata=dict(platform=platform.platform(),python=platform.python_version(),
                  packages={name:importlib.metadata.version(name) for name in ['numpy','gguf','torch']},
                  scope='Read-only audit of existing local weights. No pruning, model changes or inference quality test.')
    (a.output/'report.json').write_text(json.dumps(dict(metadata=metadata,models=models),indent=2)+'\n')
    with (a.output/'tensors.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(details[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(details)
    lines=['# Нулевые и околонулевые веса наших моделей','',
           'Анализ 2026-10-18. Веса прочитаны без изменений; прунинг и проверка качества не выполнялись.', '',
           '## Доля значений', '',
           '| Модель | Значений весов | Точно 0, % | abs ≤ 0.0001, % | abs ≤ 0.001, % | abs ≤ 0.01, % | abs ≤ 0.1 × RMS тензора, % |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for m in models:
        t={t['rule']:t for t in m['thresholds']}
        lines.append(f"| {m['model']} | {m['parameters']:,} | "+' | '.join(f"{t[k]['pct']:.4f}" for k in
                     ['exact_zero','abs_0.0001','abs_0.001','abs_0.01','rms_0.1'])+' |')
    lines += ['', '## Маска и поток ненулевых значений: расчёт, не реализованный формат', '',
              'Сохраняем масштабы и другие параметры квантования. Для каждого тензора добавляем ceil(N/8) байт '
              'маски и упаковываем оставшиеся коды с исходной битностью. FP32 считается с 32 битами на значение. '
              'Смещения, выравнивание и признаки выбора dense/sparse не учтены: это оптимистичная оценка.', '',
              '| Модель | Плотные тензоры, МиБ | Маска для всех, точные нули, МиБ | Выбор dense/sparse по тензору, МиБ | Экономия выбора, % |',
              '| --- | ---: | ---: | ---: | ---: |']
    for m in models:
        t=m['thresholds'][0]
        lines.append(f"| {m['model']} | {m['stored_tensor_bytes']/2**20:.3f} | {t['all_sparse_bytes']/2**20:.3f} | "
                     f"{t['selective_sparse_bytes']/2**20:.3f} | {t['selective_saved_pct']:.4f} |")
    lines += ['', '## Методика и ограничения', '',
              '- GGUF: восстановление значений через gguf.quants.dequantize; Supra/Stentor Q4_K_M содержат Q4_K, Q6_K и F32, Stories также Q5_0 и Q8_0. Нулевой код не приравнивается к нулевому весу.',
              '- PLE и прежняя TinyTalk: форматы читаются по локальному коду движков; FP16 и BF16 масштабы различаются. Общий embedding/output не считается повторно.',
              '- Vagh: только model_state, без optimizer. GPT-2: исключены причинные маски внимания; shared storage считается один раз. Список исключений сохранён в JSON.',
              '- Считаются хранимые значения тензоров (включая обучаемые bias/norm), а не байты токенизатора и заголовков. Stories GGUF содержит 58,073,600 значений, несмотря на имя 42M.',
              '- Проверены SHA-256 источников, конечность восстановленных значений, размеры и число параметров. Абсолютные пороги зависят от масштаба весов; относительные считаются отдельно от RMS каждого тензора.',
              '- Нули после деквантования не обязательно были нулями до квантования. Q4/Q8 — разные представления; здесь не проверяются исходные FP32 checkpoint этих GGUF.',
              '- Обнуление ненулевых значений меняет модель. Доля суммы квадратов удаляемых весов в JSON — численная характеристика, не оценка падения качества.',
              '- Формат маски потребует нового декодера/kernel. Здесь не обещается ускорение, уменьшение KV-кэша или совместимость с текущей прошивкой.', '',
              '## Воспроизведение', '', '```sh',
              "python -m pip install 'gguf==0.19.0' 'numpy==2.2.6' 'torch>=2.6'",
              'python tools/analyze_weight_sparsity.py --output build/weight-sparsity-new','```','',
              'Нужны уже скачанные модели по путям из report.json. На этом Mac использовано существующее Torch-окружение '
              'с gguf из build/compression-venv310 через PYTHONPATH. Анализ не скачивает веса.', '',
              '[JSON с источниками и оценками всех порогов](report.json), [по каждому тензору](tensors.csv).', '',
              'Реализация деквантования: [официальный gguf-py](https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/quants.py).', '']
    (a.output/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
