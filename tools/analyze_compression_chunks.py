#!/usr/bin/env python3
"""Inspect existing TMLCMP01 indexes and separate GGUF metadata/tensor regions."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct


def gguf_layout(source):
    """Read the little-endian GGUF v3 directory, without interpreting weight types."""
    pos = 0

    def take(size):
        nonlocal pos
        if size < 0 or pos + size > len(source):
            raise ValueError('Truncated GGUF')
        value = source[pos:pos + size]
        pos += size
        return value

    def number(fmt):
        return struct.unpack('<' + fmt, take(struct.calcsize('<' + fmt)))[0]

    def string():
        return take(number('Q'))

    def value(kind):
        scalar = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f',
                  7: '?', 10: 'Q', 11: 'q', 12: 'd'}
        if kind in scalar:
            return number(scalar[kind])
        if kind == 8:
            return string()
        if kind == 9:
            element, count = number('I'), number('Q')
            for _ in range(count):
                value(element)
            return None
        raise ValueError(f'Unknown GGUF metadata type {kind}')

    if take(4) != b'GGUF' or number('I') != 3:
        raise ValueError('Expected little-endian GGUF v3')
    tensors, entries = number('Q'), number('Q')
    alignment = 32
    for _ in range(entries):
        key = string()
        item = value(number('I'))
        if key == b'general.alignment':
            alignment = item
    metadata_end = pos
    tensor_offsets = []
    for _ in range(tensors):
        string()
        dimensions = number('I')
        take(8 * dimensions)
        number('I')
        tensor_offsets.append(number('Q'))
    if not isinstance(alignment, int) or alignment < 8 or alignment % 8:
        raise ValueError('Invalid GGUF alignment')
    start = (pos + alignment - 1) // alignment * alignment
    assert start < len(source) and min(tensor_offsets) == 0
    assert all(offset % alignment == 0 and start + offset < len(source) for offset in tensor_offsets)
    return dict(version=3, metadata_entries=entries, tensor_count=tensors,
                metadata_end=metadata_end, directory_end=pos, alignment=alignment,
                tensor_data_start=start, tensor_region_bytes=len(source)-start)


def write_csv(path, rows):
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    benchmark = json.loads((args.benchmark / 'results.json').read_text())
    source = args.source.read_bytes()
    assert hashlib.sha256(source).hexdigest() == benchmark['metadata']['source_sha256']
    layout = gguf_layout(source)
    records = [r for r in benchmark['results'] if r['chunk_bytes']]
    best = {}
    for row in records:
        key = (row['codec'], row['chunk_bytes'])
        if key not in best or row['file_bytes'] < best[key]['file_bytes']:
            best[key] = row
    summaries, detailed, regions = [], [], []
    for row in records:
        artifact = args.benchmark / row['artifact']
        # Hash the existing artifact to bind this index analysis to the verified benchmark.
        digest = hashlib.sha256()
        with artifact.open('rb') as stream:
            while data := stream.read(2**20):
                digest.update(data)
            assert digest.hexdigest() == row['artifact_sha256']
            stream.seek(0)
            magic, codec_id, chunk, size = struct.unpack('<8sIIQ', stream.read(24))
            assert (magic, codec_id, chunk, size) == (b'TMLCMP01', 1 if row['codec'] == 'zstd' else 2,
                                                    row['chunk_bytes'], len(source))
            count = (size + chunk - 1) // chunk
            assert count == row['chunks']
            offsets = struct.unpack(f'<{count+1}Q', stream.read(8*(count+1)))
            index_bytes = stream.tell()
            assert offsets[0] == 0 and all(b > a for a, b in zip(offsets, offsets[1:]))
            assert index_bytes + offsets[-1] == artifact.stat().st_size == row['file_bytes']
        chunks = []
        for i in range(count):
            offset = i * chunk
            original = min(chunk, size-offset)
            compressed = offsets[i+1]-offsets[i]
            region = ('metadata' if offset+original <= layout['tensor_data_start'] else
                      'tensor_data' if offset >= layout['tensor_data_start'] else 'boundary')
            chunks.append(dict(codec=row['codec'], level=row['level'], chunk_kib=chunk//1024,
                               chunk_index=i, source_offset=offset, region=region,
                               original_bytes=original, frame_bytes=compressed,
                               saved_bytes=original-compressed, saved_pct=100*(1-compressed/original)))
        ratios = sorted(c['frame_bytes']/c['original_bytes']*100 for c in chunks)
        summary = dict(codec=row['codec'], level=row['level'], chunk_kib=chunk//1024, chunks=count,
                       file_bytes=row['file_bytes'], saved_pct=row['saved_pct'],
                       smaller_chunks=sum(c['saved_bytes'] > 0 for c in chunks),
                       equal_chunks=sum(c['saved_bytes'] == 0 for c in chunks),
                       larger_chunks=sum(c['saved_bytes'] < 0 for c in chunks),
                       save_under_one_pct=sum(c['saved_pct'] < 1 for c in chunks),
                       median_size_pct=statistics.median(ratios),
                       p95_size_pct=ratios[math.ceil(.95*count)-1],
                       min_size_pct=ratios[0], max_size_pct=ratios[-1],
                       total_expansion_bytes=sum(-c['saved_bytes'] for c in chunks if c['saved_bytes'] < 0),
                       index_bytes=index_bytes)
        summaries.append(summary)
        if row is best[(row['codec'], chunk)]:
            detailed.extend(chunks)
            for region in ['metadata', 'boundary', 'tensor_data']:
                part = [c for c in chunks if c['region'] == region]
                if part:
                    original = sum(c['original_bytes'] for c in part)
                    compressed = sum(c['frame_bytes'] for c in part)
                    regions.append(dict(codec=row['codec'], level=row['level'], chunk_kib=chunk//1024,
                                        region=region, chunks=len(part), original_bytes=original,
                                        frame_bytes=compressed, saved_bytes=original-compressed,
                                        saved_pct=100*(1-compressed/original)))
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output/'all-levels.csv', summaries)
    write_csv(args.output/'best-chunks.csv', detailed)
    write_csv(args.output/'regions.csv', regions)
    (args.output/'layout.json').write_text(json.dumps(dict(source_sha256=benchmark['metadata']['source_sha256'],
                                                        **layout), indent=2)+'\n')
    lines = ['# Разбор отдельных чанков Supra Q4', '',
             'Анализ существующих 80 контейнеров: два алгоритма, уровни 1–10, четыре размера чанка. '
             'Их SHA-256 повторно сверены с исходным benchmark; повторного сжатия не было. '
             'Распаковка всех файлов проверена в [основном опыте](../README.md).', '',
             f"Количество тензоров в GGUF: {layout['tensor_count']}. Область данных тензоров начинается "
             f"с байта **{layout['tensor_data_start']:,}**; всё до неё — заголовок, метаданные, "
             'описания тензоров и выравнивание. Исходник проверен по SHA-256.', '',
             '## Лучший уровень для каждого размера', '',
             '| Алгоритм | Уровень | КиБ | Чанков | Файл, байт | Экономия % | Уменьшились | Увеличились | Медианный размер чанка, % исходного |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in summaries:
        if s['level'] == best[(s['codec'], s['chunk_kib']*1024)]['level']:
            lines.append(f"| {s['codec']} | {s['level']} | {s['chunk_kib']} | {s['chunks']} | {s['file_bytes']:,} | "
                         f"{s['saved_pct']:.3f} | {s['smaller_chunks']} | {s['larger_chunks']} | {s['median_size_pct']:.3f} |")
    lines += ['', 'Итоговый файл включает индекс. Размер отдельного чанка включает заголовок и checksum frame, '
              'но не индекс контейнера. Медиана — средний по порядку процент среди чанков; общая экономия считается по байтам. '
              'Последний чанк может быть неполным. Увеличение размера означает накладные расходы формата, '
              'а не потерю информации.', '', '## Метаданные и область весов', '',
              '| Алгоритм | КиБ | Область | Чанков | Исходные байты | Сжатые байты | Экономия % |',
              '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for r in regions:
        lines.append(f"| {r['codec']} | {r['chunk_kib']} | {r['region']} | {r['chunks']} | {r['original_bytes']:,} | "
                     f"{r['frame_bytes']:,} | {r['saved_pct']:.3f} |")
    lines += ['', '`metadata` — чанк целиком до области тензоров; `tensor_data` — целиком внутри области тензоров '
              '(она может содержать межтензорное выравнивание); `boundary` пересекает границу. '
              'Смешанный чанк не приписывается ни одной чистой области. Индекс в этой таблице не учтён. '
              'Это разбиение байтов GGUF, а не послойный формат загрузки модели.', '',
              '## Данные и повторение', '',
              '- [Все 80 вариантов: статистика, медиана, P95, расширение](all-levels.csv).',
              '- [Каждый чанк восьми лучших вариантов: смещение, размеры, область](best-chunks.csv).',
              '- [Суммы по областям](regions.csv), [граница данных GGUF](layout.json).', '',
              '```sh', 'python3 tools/analyze_compression_chunks.py \\',
              '  --benchmark build/compression-supra-q4 \\',
              '  --source build/model-playground/supra/Supra2-100M-Instruct-Q4_K_M.gguf \\',
              '  --output build/compression-chunks-new', '```', '',
              'Парсер ограничен little-endian GGUF v3. Он читает структуру по '
              '[спецификации GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md); '
              'типы и значения самих весов не интерпретируются. Новых измерений RAM, '
              'скорости декодирования или inference на ESP этот анализ не содержит.', '']
    (args.output/'README.md').write_text('\n'.join(lines))
    print(json.dumps(layout, indent=2))
    print(f'Checked {len(summaries)} artifact indexes; exported {len(detailed)} individual chunks')


if __name__ == '__main__':
    main()
