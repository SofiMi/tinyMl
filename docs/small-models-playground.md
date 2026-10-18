# Маленькие модели на компьютере

Эксперимент 2026-10-18. Это отдельные модели для Mac/Linux; прошивка ESP и GPT-2
не меняются. Модели отвечают на английском. Их ответы не заменяются правилами.

## Запуск на подготовленном Mac

Из корня репозитория:

```sh
bash tools/small_models.sh
```

Выберите номер. Можно сразу выбрать модель:

```sh
bash tools/small_models.sh stentor
bash tools/small_models.sh stories
bash tools/small_models.sh vagh
bash tools/small_models.sh supra
bash tools/small_models.sh supra-q8
bash tools/small_models.sh stentor-q8
```

| Имя | Что запускается | Размер скачанных весов |
| --- | --- | ---: |
| `stentor` | Stentor-30M-Instruct, 30 419 712 параметров, Q4_K_M | 20 913 792 байта |
| `stories` | llama2.c-stories42M, GGUF-конверсия с 58 073 600 параметрами, Q4_K_M | 39 680 864 байта |
| `vagh` | VaghLM, 42 615 808 уникальных параметров, CPU FP32 | checkpoint 511 474 827 байт, включая optimizer |
| `supra` | Supra2-100M-Instruct, 100 684 032 параметра, Q4_K_M | 67 102 400 байт |
| `supra-q8` | Та же Supra2-100M-Instruct, Q8_0 | 108 208 832 байта |
| `stentor-q8` | Та же Stentor-30M-Instruct, Q8_0 | 33 110 656 байт |

Для начала попробуйте `bash tools/small_models.sh supra`. Q8-варианты добавлены
для сравнения точности весов, а не как обещание более качественного диалога.
Чтобы сравнить Q4/Q8, задавайте одинаковые вопросы с одинаковыми параметрами.
`--temperature 0` убирает случайный выбор следующего токена.

**VaghLM не является проверенной instruct-версией:** карточка и пример автора
ссылаются на `finetuned_v2_best.pt`, но в зафиксированной публикации присутствует
только `checkpoints/best_checkpoint.pt` (step 44000, loss 1.991916732788086).
Запускаем именно доступный checkpoint; Q4 для него не создавали. Диалоговые
маркеры соответствуют коду автора, наличие самого диалогового обучения этого
checkpoint не подтверждено.

По умолчанию **каждая строка — независимый запрос**, без прошлых ответов и без
роли Ember. Для Stentor, Supra и VaghLM применяется собственный формат вопроса модели;
это отличается от системной роли персонажа. Команды:

- `/quit` или Ctrl+C — выход;
- `/new` — сброс истории;
- `/stats` — модель и число сохранённых реплик;
- `/help` — подсказка.

История включается явно, если хочется проверить несколько реплик подряд:

```sh
bash tools/small_models.sh stentor --history
bash tools/small_models.sh vagh --history
bash tools/small_models.sh supra --history
```

При заполнении контекста программа просит `/new`; молча отбрасывать старые
реплики она не должна. Stentor ограничен 512 токенами вместе с ответом, VaghLM —
2048. Режим истории VaghLM экспериментальный: официальный пример однорепличный.
У Supra config также разрешает 2048, но автор сообщает об обучении только на
1024 токенах и предупреждает о непроверенной экстраполяции выше этого предела.
Для Supra применяется оригинальный ChatML-шаблон без дополнительного BOS;
генерация останавливается на `</s>` или `<|im_end|>`.

Для Stories вводите начало истории, например:

```text
Once upon a time, a baby dragon met a little boy named Ramil.
```

Stories продолжает текст и не является чат-ассистентом. У неё нет истории между
запросами. Для других моделей также доступно продолжение чистого текста:

```sh
bash tools/small_models.sh vagh --raw
```

Дополнительные параметры: `--max-tokens 128`, `--temperature 0.7`, `--threads 2`.
`--temperature 0` включает greedy. Можно записать точные входы, ID токенов,
настройки и ответы в новый файл:

```sh
bash tools/small_models.sh stentor --trace build/model-playground/my-stentor.jsonl
```

## Установка на другом компьютере

Нужны Python 3.10–3.12, C/C++ compiler и доступ в интернет при установке.
На Linux обычно требуются `build-essential`, `cmake`, `python3-venv`;
на Mac — Xcode Command Line Tools. Сам чат после установки работает офлайн.

```sh
python3 -m venv build/model-playground/venv
build/model-playground/venv/bin/python -m pip install 'torch>=2.6' 'tokenizers>=0.21' 'sentencepiece>=0.2' 'jinja2>=3.1'
build/model-playground/venv/bin/python -m pip install 'llama-cpp-python==0.3.16' --config-settings=cmake.define.GGML_NATIVE=OFF
python3 tools/small_models.py fetch all
bash tools/small_models.sh
```

`fetch` проверяет свободный диск, фиксированную ревизию, размеры, SHA-256 для
LFS и Git blob ID для обычных файлов. Модели лежат в `build/model-playground/`,
не попадают в Git. Файл `tools/small_models.lock.json` фиксирует источники.
Повторный `fetch` пропускает уже проверенные файлы; повреждённые существующие
файлы не затираются автоматически. Оставьте примерно 2–3 ГиБ для зависимостей,
сборки и моделей; PyTorch на отдельных платформах может потребовать больше.

Обёртка выбирает `build/model-playground/venv`, затем существующее окружение
`~/.cache/tinyml-eval-venv`, затем `python3`. Переопределение:
`SMALL_MODELS_PYTHON=/path/to/python bash tools/small_models.sh stentor`.

## Токенизация и проверка

В опубликованном Stentor GGUF специальные `[INST]` / `[/INST]` имеют тип NORMAL.
При первоначальном запуске через Ollama 0.40.5 получалось другое разбиение
промпта. Поэтому конечный запуск использует **исходный HF tokenizer.json**,
официальный `chat_template.jinja` и передаёт готовые ID непосредственно в
`llama_cpp.Llama.generate`. Ответ декодируется тем же HF-токенизатором.
Все соответствия ID → текст сверяются между HF и GGUF перед генерацией.
Скачанные Q4-файлы остаются неизменными. Сервера и Ollama для запуска не нужны.

VaghLM использует проверенные закреплённые модули автора, SentencePiece и
`torch.load(..., weights_only=True)` с `strict=True` для state_dict.

Проверка настоящих CLI и генерации:

```sh
build/model-playground/venv/bin/python tools/check_small_models.py --output build/model-playground/check-new
```

Выходной каталог должен быть новым. На подготовленном Mac вместо этого Python
можно использовать `~/.cache/tinyml-eval-venv/bin/python`.
Параметр `--models supra supra-q8 stentor-q8` ограничивает CLI-проверку этими
вариантами. Без него проверяются все варианты из lock-файла.
Проверка входных ID, управления историей и получения ответов **не доказывает
качество разговора и не является измерением скорости ESP**.

На Mac вычисления настроены на CPU (`n_gpu_layers=0`); установленная библиотека
также содержит Metal. Проверки нативного движка выполнялись вне песочницы.
Зафиксированные результаты и ограничения качества:
[локальные проверки](evaluation/small-models/README.md).

## Меньше параметров, больше битов?

Это отдельный компромисс: число параметров определяет размер обученной модели,
а число битов — точность хранения её весов. Квантование может ухудшать ответы,
но повышение точности не добавляет знаний и не исправляет недостаточное обучение.
Даже прежняя VaghLM в FP32 путала роли.

Без накладных расходов 100M × 4 бита, 50M × 8 бит и 25M × 16 бит занимают
одинаковые 50 MB. Реальные файлы отличаются: учитываются масштабы, токенизатор,
метаданные и тензоры большей точности; RAM/KV-cache считаются отдельно.

Полезно сначала сравнить **один checkpoint Q4 против Q8**, затем разные модели.
Так не смешиваются потери квантования и различия архитектуры/обучения.
Stentor Q8 и Supra Q4 имеют разные размеры файлов; это не эксперимент с равным
бюджетом памяти. Повторить небольшой опыт на одинаковых вопросах:

```sh
build/model-playground/venv/bin/python tools/compare_small_models.py --output build/model-playground/precision-new
```

Сохраняются сырые ответы при greedy: по три независимых запроса и одному
шестирепличному разговору с собственными ответами каждого варианта. Это небольшой
диагностический опыт, а не независимый benchmark. [Результаты](evaluation/supra-precision/README.md).

Отдельно проверено сжатие всего Supra Q4 без потерь: zstd/LZ4, уровни 1–10,
целиком и независимыми чанками 64/128/254/512 КиБ. [Все размеры и времена,
CSV и методика](evaluation/supra-compression/README.md). Это упаковка файла,
а не новая квантованная модель или готовый загрузчик для ESP.

Источники: [VaghLM](https://huggingface.co/Heet24/model42M-v1),
[Stories42M Q4](https://huggingface.co/mradermacher/llama2.c-stories42M-GGUF),
[Stentor Q4](https://huggingface.co/mradermacher/Stentor-30M-Instruct-GGUF),
[исходный Stentor](https://huggingface.co/StentorLabs/Stentor-30M-Instruct).
Supra: [исходная модель](https://huggingface.co/SupraLabs/Supra2-100M-Instruct),
[закреплённые Q4/Q8](https://huggingface.co/liodon-ai/Supra2-100M-Instruct-imatrix-GGUF).
