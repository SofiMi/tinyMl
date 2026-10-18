# PLE TinyStories из slvDev/esp32-ai

Перенесено 2026-10-18 из [slvDev/esp32-ai](https://github.com/slvDev/esp32-ai/tree/52ebcbbd905270592c1d415efd1128d04d0fa365),
commit `52ebcbbd905270592c1d415efd1128d04d0fa365`.

Файлы в `upstream/` скопированы без изменений:

| Здесь | В исходном репозитории |
| --- | --- |
| `upstream/llm.h` | `runtime/llm.h` |
| `upstream/bpe_tokenizer.h` | `runtime/bpe_tokenizer.h` |
| `upstream/host_verify/staging_verify.c` | `runtime/host_verify/staging_verify.c` |
| `upstream/host_verify/tokenizer_conformance.c` | `runtime/host_verify/tokenizer_conformance.c` |
| `upstream/tools/generate_tokenizer_header.py` | `firmware/esp32_barista/tools/generate_tokenizer_header.py` |

Распределение памяти и двухъядерный worker в `main/console.cpp` адаптированы из
`firmware/esp32_tinystories/esp32_tinystories.ino`. Arduino заменён ESP-IDF;
вместо фиксированного ввода добавлены кодирование текста и UART0-консоль.
MIT-лицензия исходного кода сохранена в `LICENSE`.

Веса и tokenizer: [slvDev/esp32-ai-tinystories](https://huggingface.co/slvDev/esp32-ai-tinystories).
Скачаны из `resolve/main`, но идентичность фиксируется размером и SHA-256,
совпадающими с `scripts/fetch_model.sh` указанного commit:

| Файл | Байт | SHA-256 |
| --- | ---: | --- |
| `assets/model.bin` | 14912348 | `1d8326c05c383ccfa615f5455575802817cb453dbc7ab28875d41a9dbb45477e` |
| `assets/tokenizer.json` | 1788896 | `4e28163669f2249af31a528a54fc25064dcbd0a34edbfa7bedb16d2d600ec7ae` |
| `assets/metadata.json` | 2388 | `0b711aafb608e8652974d64adea5b5f80499a970f7e66e3747111a922bbd4017` |

`tools/prepare_esp32ai.py` проверяет веса и tokenizer, сверяет metadata и строит
таблицы без сторонних Python-зависимостей. Для TinyStories специальный EOS
с id=0 используется только на выходе. Его буквальное написание
`<|endoftext|>` отвергается консолью; обычный ASCII проходит upstream BPE.
Декодер хранит исходные байты токенов, сохраняя части UTF-8 между токенами.

Согласно metadata релиза: веса и tokenizer — MIT, обучающие данные TinyStories —
CDLA-Sharing-1.0. Исходный обучающий checkpoint не опубликован; точная ревизия
среза обучающих данных upstream не зафиксирована. Здесь новых весов не обучали.

Проверки переноса, ограничения и замеры: [docs/esp32ai-port.md](../../docs/esp32ai-port.md).
