# Vendored TinyTalk engine and artifacts

Source: https://github.com/therezor/cardputer-ai

Revision: `78c5128672b750977619dc0a6c3b8baed91168ed`.

Copied without modification:

- `main/llm.cpp` → `llm.cpp`
- `main/llm.h` → `include/llm.h`
- `main/dot_q4_pie.S` → `dot_q4_pie.S`
- `embed/model_neo_q4.bin` → `assets/model_neo_q4.bin`
- `embed/tok_neo.bin` → `assets/tok_neo.bin`
- `LICENSE` and `NOTICE.md`
- `tools/host/esp_heap_caps.h` and `tools/host/freertos/*.h` → project `tools/host/`

The local CMake/assembly wrapper embeds the binaries directly in aligned
read-only flash sections, replacing upstream's generated C++ byte arrays.
The USB console and SSD1306 integration are specific to this project.

SHA-256:

```text
183fe4c212fd0fd17a8e048ee3eab4a0095a039df290f0c58cafff70ca17fe2a  model_neo_q4.bin
7b915b9779c12fa896ad839a81ce315c757419e71a4dee7f7890d2578051429f  tok_neo.bin
```

The original NOTICE is retained verbatim, including references to upstream
paths and components not copied here. Its base-model description predates
some TinyTalk revisions; it is not a verified parameter count for this binary.
Consult that notice for model/data provenance separately from the MIT code license.
