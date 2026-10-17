# Проверка разговорных моделей с системным описанием персонажа

17 октября 2026 года. По решению пользователя персонажа не закрепляем в весах:
модель должна уметь общий диалог, а имя и характер получать из system prompt.
Нового обучения, pruning и замены прошивки в этом опыте не было.

## Что проверено

Использован `tools/dragon_hf.py`, оригинальный chat template каждого checkpoint,
FP32-вычисления на CPU Mac, greedy, максимум 48 выходных токенов и бюджет
контекста 1024. Каждый многоходовый тест использует собственные предыдущие
ответы модели. `heldout-v2.json` содержит 20 сценариев / 29 ответов и теперь
является development-набором, а не независимым финальным тестом.

| Модель | Параметры | Источник, закреплённая revision | Проверки |
|---|---:|---|---|
| SmolLM2-135M-Instruct | 134 515 008 | [HuggingFaceTB](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac) | Подробная и короткая роль, затем отдельный нейтральный sanity check |
| MiniMind2-Small | 25 829 888 | [jingyaogong](https://huggingface.co/jingyaogong/MiniMind2-Small/tree/8c0c0de640cd532fee03d329b95a58c56591b5bc) | Подробная роль на тех же английских вопросах |

У MiniMind шаблон находится в отдельном `chat_template.jinja`; он загружен из
той же revision. При отсутствии шаблона первый запуск остановился с ошибкой
до генерации. Сохранённые ответы получены после загрузки оригинального файла,
без придуманного формата диалога и без исполнения стороннего Python-кода.

## Результаты

**Ни один проверенный вариант пока не годится как надёжный дракон.** Это вывод
по данным разговорам, а не утверждение обо всех возможных настройках моделей.

- SmolLM2 с подробной ролью на вопрос о своём имени отвечает, что имени у неё
  ещё нет. После `I am Hugo` присваивает себе имя Hugo; на просьбу различить
  имена пишет `My name is Ember, and my name is Hugo.` Короткая роль не устраняет
  эту путаницу. У модели есть удачные отдельные ответы, например четыре ноги у
  собаки и применение коррекции имени до Iris.
- В нейтральном режиме SmolLM2 может объяснять гравитацию, но после представления
  пользователя Sofia снова присваивает имя себе. Поэтому проблема не ограничена
  исключительно описанием дракона. Это небольшой ad hoc контроль, не benchmark.
- MiniMind2 на `My name is Elena` просит предоставить вопрос; на просьбу назвать
  тип животного просит дополнительные данные. Английские ответы часто шаблонны
  или повторяются, роль и связи между именами не удерживаются. Китайский в этом
  опыте не проверялся: вывод относится к нашей английской задаче.

SmolLM2 завершила по EOS 20/29 ответов с подробной ролью и 25/29 с короткой;
MiniMind — 18/29. Остальные достигли лимита. EOS — свойство формата, не качества.
В JSON сохранены старые keyword smoke-checks: они принимают и ошибочную фразу
`My name is Hugo` за совпадение имени пользователя. Их нельзя называть accuracy.
Разные токенизаторы также дают разную длину текста при одинаковых 48 токенах.

Полные результаты:

- [SmolLM2, подробная роль](evaluation/dialogue-candidates/smollm2-full.json)
- [SmolLM2, короткая роль](evaluation/dialogue-candidates/smollm2-compact.json)
- [SmolLM2, нейтральный контроль](evaluation/dialogue-candidates/smollm2-assistant-controls.json)
- [MiniMind2, подробная роль](evaluation/dialogue-candidates/minimind2-full.json)

Свежий набор `tests/dragon/compression-v1.json` и его рубрика подготовлены до
просмотра этих результатов, но пока **не запускались**. Сохранить их для следующей
выбранной модели; после просмотра они тоже перестанут быть независимыми.

## Повторить или поговорить локально

Используется окружение из [dragon-training.md](dragon-training.md) с
`tools/requirements-dragon.txt`. Веса не включены в Git. В активированном
Python-окружении скачать только необходимые файлы:

```python
from pathlib import Path
from huggingface_hub import snapshot_download

candidates = [
    ("HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac", "tinyml-smollm2-135m-instruct"),
    ("jingyaogong/MiniMind2-Small", "8c0c0de640cd532fee03d329b95a58c56591b5bc", "tinyml-minimind2-small"),
]
for model_id, revision, directory in candidates:
    snapshot_download(model_id, revision=revision,
        allow_patterns=["config.json", "generation_config.json", "model.safetensors",
                        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                        "*.jinja", "README.md", "README_en.md"],
        local_dir=Path.home()/".cache"/directory)
```

Чат на настроенном Mac (другая среда — через `DRAGON_PYTHON`):

```sh
bash tools/dragon_chat.sh minimind
bash tools/dragon_chat.sh smollm
```

`/new` очищает историю, `/quit` завершает. Чат не использует сеть.

Воспроизводимый тест MiniMind из корня проекта:

```sh
python tools/dragon_hf.py eval \
  --model "$HOME/.cache/tinyml-minimind2-small" \
  --source-revision 8c0c0de640cd532fee03d329b95a58c56591b5bc \
  --persona full --output build/dragon/minimind2-full.json
```

Для SmolLM2 default-путь уже установлен; `--persona compact` включает короткое
описание, `--persona assistant` — нейтрального помощника SmolLM. Последний
профиль предназначен для контроля SmolLM, а не для оценки идентичности иных моделей.

## Что меняется в следующем опыте

[Обзор обучения диалогу](research/dialogue-instruction-papers.md),
[современная дистилляция](research/modern-distillation-papers.md) и
[сохранение навыков](research/retention-papers.md) объясняют выбор метода.
Следующая обучающая смесь должна варьировать system prompt и учить общий диалог,
исправление фактов и различение собеседников. Старые постоянные ответы Эмбера
не являются новой основной целью обучения.

MiniMind ближе по размеру к плате, но её текущий английский недостаточен.
SmolLM2 существенно больше бюджета flash и тоже не прошла эту проверку качества.
Поэтому ни одну пока не выбираем для сжатия и переноса. Внешние заявления о
скорости двигателя не заменяют проверку диалоговой модели; результаты на CPU
Mac не являются измерением скорости ESP32.
