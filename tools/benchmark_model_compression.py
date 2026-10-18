#!/usr/bin/env python3
"""Lossless whole-file and independently indexed chunk compression benchmark."""
import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
import shutil
import statistics
import struct
import time

import lz4
import lz4.frame
import zstandard as zstd

HEADER = struct.Struct('<8sIIQ')
MAGIC = b'TMLCMP01'
CHUNKS = (0, 64 * 1024, 128 * 1024, 254 * 1024, 512 * 1024)


def encode(codec, level, blocks):
    if codec == 'zstd':
        compressor = zstd.ZstdCompressor(level=level, threads=0,
                                        write_checksum=True, write_content_size=True)
        return [compressor.compress(block) for block in blocks]
    return [lz4.frame.compress(block, compression_level=level,
                              block_size=lz4.frame.BLOCKSIZE_MAX4MB,
                              block_linked=True, content_checksum=True,
                              block_checksum=False, store_size=True) for block in blocks]


def decode(codec, frames):
    decompress = zstd.ZstdDecompressor().decompress if codec == 'zstd' else lz4.frame.decompress
    return [decompress(frame) for frame in frames]


def write_artifact(path, codec, chunk_size, source_size, frames):
    with path.open('xb') as stream:
        if chunk_size:
            stream.write(HEADER.pack(MAGIC, 1 if codec == 'zstd' else 2, chunk_size, source_size))
            offset = 0
            for frame in frames:
                stream.write(struct.pack('<Q', offset))
                offset += len(frame)
            stream.write(struct.pack('<Q', offset))
        for frame in frames:
            stream.write(frame)


def verify_artifact(path, codec, chunk_size, source_size, expected_sha):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        if not chunk_size:
            digest.update(decode(codec, [stream.read()])[0])
        else:
            magic, codec_id, stored_chunk, stored_size = HEADER.unpack(stream.read(HEADER.size))
            assert (magic, codec_id, stored_chunk, stored_size) == (
                MAGIC, 1 if codec == 'zstd' else 2, chunk_size, source_size)
            count = (source_size + chunk_size - 1) // chunk_size
            offsets = struct.unpack(f'<{count + 1}Q', stream.read(8 * (count + 1)))
            payload_start = stream.tell()
            assert offsets[0] == 0 and payload_start + offsets[-1] == path.stat().st_size
            for i in range(count):
                assert offsets[i + 1] > offsets[i]
                # Each seek/decode starts independently, without a previous chunk or dictionary.
                stream.seek(payload_start + offsets[i])
                block = decode(codec, [stream.read(offsets[i + 1] - offsets[i])])[0]
                assert len(block) == min(chunk_size, source_size - i * chunk_size)
                digest.update(block)
    if digest.hexdigest() != expected_sha:
        raise RuntimeError(f'Round-trip SHA-256 mismatch: {path}')


def markdown(meta, rows):
    lines = ['# Сжатие Supra2 100M Q4: zstd и LZ4', '',
             f"Дата UTC: {meta['date_utc']}. Весь исходный GGUF: **{meta['source_bytes']:,} байт** "
             f"({meta['source_bytes'] / 2**20:.3f} МиБ).", '',
             f"SHA-256: `{meta['source_sha256']}`.", '',
             '## Методика', '',
             '- Сжимается весь GGUF, включая токенизатор, метаданные и тензоры; это не дополнительное квантование.',
             '- Уровни 1–10 обоих алгоритмов. У LZ4 уровни 1 и 2 используют одинаковый быстрый режим; с 3 начинается HC.',
             '- Размеры чанков — 64/128/**254**/512 КиБ, 1 КиБ = 1024 байта. 254 не заменено на 256.',
             '- Каждый чанк — отдельный frame без общего словаря и зависимости от предыдущего чанка.',
             '- Оба формата сохраняют размер исходных данных и checksum. LZ4: linked blocks внутри frame, максимум блока 4 МиБ.',
             '- Целый файл — обычный .zst/.lz4. Чанки — контейнер .tmlc: 24 байта заголовка и (N+1) × 8 байт индекса, затем frames.',
             '- В таблицах размер фактического файла **включает заголовки frames и индекс контейнера**.',
             f"- Время — медиана {meta['repeats']} последовательных проходов в памяти, включая создание codec context и Python-обёртку; "
             'без дискового I/O, хеширования и сравнения результата. Один поток, без параллельных запусков этого benchmark.',
             '- Все повторы распакованы и побайтно сверены; каждый сохранённый файл отдельно прочитан с диска и проверен SHA-256.',
             '- Это измерения Mac, не ESP. Скорости зависят от машины и текущей нагрузки; небольшие различия времени незначимы.',
             '', f"Среда: `{meta['platform']}`, Python {meta['python']}, "
             f"zstandard {meta['zstandard_python']} / libzstd {meta['libzstd']}, "
             f"lz4 {meta['lz4_python']} / liblz4 {meta['liblz4']}.", '',
             'Размер % = размер результата / исходный размер × 100. Экономия % = 100 − размер %.',
             'Скорость распаковки = исходные МиБ / секунды распаковки. Все замеры также есть в [CSV](results.csv) и [JSON](results.json).', '']
    for chunk in CHUNKS:
        lines += [f"## {'Целый файл' if chunk == 0 else f'Чанки {chunk // 1024} КиБ'}", '',
                  '| Алгоритм | Уровень | Файл, байт | МиБ | Размер % | Экономия % | Сжатие, с | Распаковка, с | Распаковка, МиБ/с |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for r in rows:
            if r['chunk_bytes'] == chunk:
                lines.append(f"| {r['codec']} | {r['level']} | {r['file_bytes']:,} | {r['file_bytes']/2**20:.3f} | "
                             f"{r['size_pct']:.3f} | {r['saved_pct']:.3f} | {r['compress_seconds']:.4f} | "
                             f"{r['decompress_seconds']:.4f} | {r['decompress_mib_s']:.1f} |")
        lines.append('')
    lines += ['## Лучшие размеры', '', '| Режим | Алгоритм | Уровень | Файл, байт | Экономия % |',
              '| --- | --- | ---: | ---: | ---: |']
    for chunk in CHUNKS:
        for codec in ('zstd', 'lz4'):
            r = min((r for r in rows if r['codec'] == codec and r['chunk_bytes'] == chunk), key=lambda r: r['file_bytes'])
            lines.append(f"| {'Целый' if not chunk else str(chunk//1024)+' КиБ'} | {codec} | {r['level']} | {r['file_bytes']:,} | {r['saved_pct']:.3f} |")
    lines += ['', '## Ограничения для ESP', '',
              'Размер сжатого файла — расход накопителя, не измерение RAM. Для вычислений веса нужно распаковывать; '
              'чанки позволяют делать это частями, но требуют отдельного загрузчика и интеграции с inference kernel. '
              'Текущий GGUF-движок не читает этот экспериментальный контейнер.', '',
              'Размер чанка не является полным бюджетом RAM: дополнительно нужны входной буфер, состояние декодера, '
              'вычислительные буферы, активации и KV-кэш. Их использование на ESP и влияние на токены/с здесь не измерены.', '',
              '## Повторение', '', '```sh',
              'python3 -m venv build/compression-venv310',
              "build/compression-venv310/bin/pip install 'zstandard==0.25.0' 'lz4==4.4.5'",
              'build/compression-venv310/bin/python tools/benchmark_model_compression.py \\',
              '  --input build/model-playground/supra/Supra2-100M-Instruct-Q4_K_M.gguf \\',
              '  --output build/compression-supra-new --repeats 3', '```', '',
              'Выходной каталог должен быть новым. Артефакты сохраняются в его подкаталоге `artifacts/`; '
              'их имена и SHA-256 перечислены в CSV/JSON. В Git передаются скрипт и отчёт, без гигабайтов архивов.', '',
              'Формат .tmlc: little-endian `<8sIIQ>` = magic `TMLCMP01`, codec (1=zstd, 2=lz4), '
              'размер чанка, исходный размер. Число чанков N вычисляется округлением вверх; далее N+1 '
              'uint64-смещений относительно начала payload. Последнее смещение равно длине payload.', '',
              'Документация: [LZ4 frame](https://python-lz4.readthedocs.io/en/stable/lz4.frame.html), '
              '[zstandard](https://python-zstandard.readthedocs.io/en/latest/compressor.html).', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    source = args.input.read_bytes()
    if not source:
        parser.error('empty input')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.output.parent).free < len(source) * 105 + 2**30:
        parser.error('need space for 100 artifacts plus 1 GiB reserve')
    args.output.mkdir(exist_ok=False)
    (args.output / 'artifacts').mkdir()
    meta = dict(date_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                source=str(args.input), source_bytes=len(source), source_sha256=hashlib.sha256(source).hexdigest(),
                platform=platform.platform(), machine=platform.machine(), python=platform.python_version(),
                zstandard_python=zstd.__version__, libzstd='.'.join(map(str, zstd.ZSTD_VERSION)),
                lz4_python=lz4.__version__, liblz4=lz4.library_version_string(), repeats=args.repeats)
    (args.output / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    rows = []
    with (args.output / 'progress.jsonl').open('x') as progress:
        for chunk_size in CHUNKS:
            step = chunk_size or len(source)
            blocks = [source[i:i + step] for i in range(0, len(source), step)]
            for codec in ('zstd', 'lz4'):
                # Untimed warm-up for this codec/chunk shape; no dictionary reuse.
                assert decode(codec, encode(codec, 1, blocks[:1])) == blocks[:1]
                for level in range(1, 11):
                    c_times, d_times = [], []
                    frame_lengths = None
                    for _ in range(args.repeats):
                        started = time.perf_counter()
                        frames = encode(codec, level, blocks)
                        c_times.append(time.perf_counter() - started)
                        lengths = [len(f) for f in frames]
                        if frame_lengths is not None:
                            assert lengths == frame_lengths
                        frame_lengths = lengths
                        started = time.perf_counter()
                        restored = decode(codec, frames)
                        d_times.append(time.perf_counter() - started)
                        assert restored == blocks
                        del restored
                    stem = f'{codec}-{level:02d}-' + (f'{chunk_size//1024}KiB.tmlc' if chunk_size else f'whole.{"zst" if codec == "zstd" else "lz4"}')
                    path = args.output / 'artifacts' / stem
                    write_artifact(path, codec, chunk_size, len(source), frames)
                    verify_artifact(path, codec, chunk_size, len(source), meta['source_sha256'])
                    size = path.stat().st_size
                    row = dict(codec=codec, level=level, chunk_bytes=chunk_size, chunks=len(blocks),
                               file_bytes=size, frame_bytes=sum(frame_lengths), index_bytes=size-sum(frame_lengths),
                               largest_frame_bytes=max(frame_lengths), size_pct=size/len(source)*100,
                               saved_pct=(1-size/len(source))*100, compress_seconds=statistics.median(c_times),
                               decompress_seconds=statistics.median(d_times),
                               decompress_mib_s=len(source)/2**20/statistics.median(d_times),
                               compress_samples=c_times, decompress_samples=d_times,
                               artifact=str(path.relative_to(args.output)),
                               artifact_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), verified=True)
                    rows.append(row)
                    progress.write(json.dumps(row) + '\n')
                    progress.flush()
                    print(f'{len(rows)}/100 {stem}: {size:,} bytes, saved {row["saved_pct"]:.2f}%, verified', flush=True)
    (args.output / 'results.json').write_text(json.dumps(dict(metadata=meta, results=rows), indent=2) + '\n')
    with (args.output / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'README.md').write_text(markdown(meta, rows))


if __name__ == '__main__':
    main()
