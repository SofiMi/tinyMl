// Offline Ember character; TinyTalk is an optional, separate research mode.
#include "llm.h"
#include "display.h"
#include "dragon.h"
#include "dragon_store.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#include "esp_psram.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
extern "C" const uint8_t MODEL_DATA[];
extern "C" const size_t MODEL_DATA_LEN;
extern "C" const uint8_t TOKENIZER_DATA[];
extern "C" const size_t TOKENIZER_DATA_LEN;
#endif

static Transformer model;
static Tokenizer tokenizer;
static Sampler sampler;
static bool raw_mode = false;
static bool model_ready = false;
static bool model_attempted = false;
static Dragon dragon;
static DragonStore dragon_store;
static std::vector<std::pair<std::string, std::string>> history;
static constexpr int kContext = 72; // Keep upstream's verified SRAM layout.
static constexpr int kMaxReply = 48;
static constexpr size_t kMaxInput = 256;

static int64_t micros() {
#ifdef ESP_PLATFORM
    return esp_timer_get_time();
#else
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}

static void yield_cpu() {
#ifdef ESP_PLATFORM
    // Let idle tasks run between model positions; retain the task watchdog.
    vTaskDelay(1);
#endif
}

static void stats() {
    printf("[mode] %s; facts: %s\n", raw_mode ? "raw TinyTalk" : "Ember (prepared replies)", dragon_store.status().c_str());
    if (model_ready) {
        printf("[model] TinyTalk 2, vocab=%d, KV=%d, temperature=%.2f, history=%zu\n",
               model.config.vocab_size, model.kv_seq_len, sampler.temperature, history.size());
    }
#ifdef ESP_PLATFORM
    printf("[memory] PSRAM=%zu, free PSRAM=%zu, free internal=%zu bytes\n",
           esp_psram_get_size(), heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
#endif
}

static std::vector<int> encode(const std::string& text) {
    // Byte BPE cannot produce more tokens than input bytes, plus BOS/EOS.
    std::vector<int> tokens(text.size() + 8);
    int count = 0;
    llm_encode(&tokenizer, text.c_str(), 0, 0, tokens.data(), &count);
    tokens.resize(count);
    return tokens;
}

static void answer(const std::string& question) {
    const int budget = model.kv_seq_len - 8;
    std::vector<int> prompt = encode("User: " + question + "\nBot:");
    if (prompt.empty() || prompt.size() > static_cast<size_t>(budget)) {
        printf("[error] Question exceeds the %d-token prompt budget. Please shorten it.\n", budget);
        return;
    }
    const auto newline = encode("\n");
    for (auto it = history.rbegin(); it != history.rend(); ++it) {
        auto segment = encode("User: " + it->first + "\nBot:" + it->second);
        segment.push_back(tokenizer.eos_id);
        segment.insert(segment.end(), newline.begin(), newline.end());
        if (segment.size() + prompt.size() > static_cast<size_t>(budget)) break;
        prompt.insert(prompt.begin(), segment.begin(), segment.end());
    }

    printf("[reading %zu tokens]\nBot:", prompt.size());
    display_begin_reply();
    fflush(stdout);
    const int64_t started = micros();
    int64_t generation_started = started;
    int token = prompt[0], slot = 0, outputs = 0, steps = 0;
    bool pending_newline = false;
    const char* stop = "position limit";
    std::string reply;
    char scratch[256];
    for (int position = 0; position < model.config.seq_len - 1; ++position) {
        if (slot >= model.kv_seq_len - 1) {
            const int removed = llm_kv_slide(&model,
                std::min<int>(model.kv_seq_len / 4, prompt.size()), model.kv_seq_len / 4);
            if (removed < 1) { stop = "context limit"; break; }
            slot -= removed;
        }
        if (position == static_cast<int>(prompt.size()) - 1) generation_started = micros();
        float* logits = llm_forward_at(&model, token, slot, position);
        ++slot;
        yield_cpu();
        if (position < static_cast<int>(prompt.size()) - 1) {
            token = prompt[position + 1];
            continue;
        }
        if (steps == 0) display_status("Thinking...");
        const int next = llm_sample(&sampler, logits);
        ++steps;
        if (next == tokenizer.eos_id) { stop = "EOS"; break; }
        const char* piece = llm_decode(&tokenizer, token, next, scratch, sizeof(scratch));
        token = next;
        ++outputs;
        if (piece && *piece) {
            if (pending_newline) {
                if (strncmp(piece, "User", 4) == 0) { stop = "next speaker"; break; }
                putchar('\n');
                reply += '\n';
                display_append("\n");
                pending_newline = false;
            }
            if (strcmp(piece, "\n") == 0) {
                pending_newline = true;
            } else {
                fputs(piece, stdout);
                reply += piece;
                display_append(piece);
                fflush(stdout);
            }
        }
        if (outputs >= kMaxReply) { stop = "reply limit"; break; }
    }
    const int64_t finished = micros();
    display_status("Ready / USB input");
    const double generation_seconds = (finished - generation_started) / 1000000.0;
    printf("\n[done] %d tokens, read=%.2fs, generation=%.2fs, %.2f steps/s, stop=%s\n",
           outputs, (generation_started - started) / 1000000.0, generation_seconds,
           generation_seconds > 0 ? steps / generation_seconds : 0.0, stop);
    if (!reply.empty()) {
        if (history.size() == 4) history.erase(history.begin());
        history.emplace_back(question, reply);
    }
}

static void help() {
    puts("Type a short English message and press Enter.\n"
         "Commands: /new (keep facts), /memory, /forget (erase facts), /stats, /quit, /help\n"
         "Examples: My name is Sofia. | I like green. | What is my name? | 1 + 1? | Tell me a story.\n"
         "/mode raw: experimental TinyTalk; /temp 0..2 controls it.\n"
         "/mode dragon: return to Ember's limited, prepared English replies.");
}

static bool read_line(std::string& line) {
    static bool after_cr = false;
    line.clear();
    bool overflow = false;
    while (true) {
        const int c = getchar();
        if (c == EOF) {
#ifdef ESP_PLATFORM
            clearerr(stdin);
            vTaskDelay(pdMS_TO_TICKS(10));
            continue;
#else
            return false;
#endif
        }
        if (after_cr && c == '\n') { after_cr = false; continue; }
        after_cr = c == '\r';
        if (c == '\n' || c == '\r') {
            putchar('\n');
            if (overflow) {
                puts("[error] Input too long; discarded. Maximum 256 bytes.");
                line.clear();
            }
            return true;
        }
        if (c == 8 || c == 127) {
            if (!overflow && !line.empty()) { line.pop_back(); fputs("\b \b", stdout); }
        } else if (c == 3) {
            line.clear();
            puts("^C");
            return true;
        } else if (c >= 32) {
            if (line.size() == kMaxInput) overflow = true;
            if (!overflow) { line += static_cast<char>(c); putchar(c); }
        }
        fflush(stdout);
    }
}

static bool run_console(const uint8_t* weights, size_t weights_size,
                        const uint8_t* vocab, size_t vocab_size, uint64_t seed) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    puts("\nEmber / offline dragon / ESP32-S3 N16R8");
    const auto init_model = [&]() {
        if (model_ready) return true;
        if (model_attempted) { puts("[error] Raw model unavailable; restart to retry."); return false; }
        model_attempted = true;
        if (!llm_init_embedded(&model, weights, weights_size, kContext) ||
            model.config.arch != ARCH_GPTNEO ||
            !llm_tokenizer_from_memory(&tokenizer, vocab, vocab_size, model.config.vocab_size) ||
            tokenizer.vocab_size != model.config.vocab_size || tokenizer.style != ARCH_GPTNEO) {
            puts("[error] Model/tokenizer initialization failed.");
            return false;
        }
        llm_build_sampler(&sampler, model.config.vocab_size, 0.8f, 0.9f, seed);
        model_ready = true;
        return true;
    };
    DragonMemory saved;
    if (dragon_store.load(saved)) dragon.restore(saved);
    else printf("[warning] %s\n", dragon_store.status().c_str());
    if (raw_mode && !init_model()) return false;
    display_init();
    display_status("Ready / USB input");
    stats();
    help();
    puts("[READY]");
    std::string line;
    while (true) {
        fputs("You> ", stdout);
        if (!read_line(line)) break;
        if (line.empty()) continue;
        if (line == "/quit") break;
        if (line == "/mode raw") {
            if (init_model()) { raw_mode = true; history.clear(); puts("[raw TinyTalk; dragon facts are isolated]"); }
        }
        else if (line == "/mode dragon") {
            raw_mode = false;
            dragon.new_conversation();
            puts("[Ember: prepared replies and saved facts]");
        }
        else if (line == "/memory") puts(dragon.describe_memory().c_str());
        else if (line == "/forget") {
            dragon.forget();
            if (dragon_store.save(dragon.memory(), true)) puts("[dragon facts cleared]");
            else printf("[warning] Facts cleared in RAM only; old saved facts may return after restart. %s\n", dragon_store.status().c_str());
        }
        else if (line == "/new") {
            history.clear();
            dragon.new_conversation();
            display_begin_reply();
            display_status("History cleared");
            puts("[history cleared; dragon facts kept]");
        }
        else if (line == "/stats") stats();
        else if (line == "/help") help();
        else if (line.compare(0, 6, "/temp ") == 0) {
            if (!raw_mode) { puts("[error] /temp applies only to /mode raw."); continue; }
            char* end = nullptr;
            const float value = strtof(line.c_str() + 6, &end);
            if (end == line.c_str() + 6 || *end || !std::isfinite(value) || value < 0 || value > 2)
                puts("[error] Use /temp followed by a number from 0 to 2.");
            else { sampler.temperature = value; stats(); }
        } else if (line[0] == '/') {
            puts("[error] Unknown command. Use /help.");
        } else if (std::any_of(line.begin(), line.end(), [](unsigned char c) { return c >= 128; })) {
            puts("[error] Please use short English messages (ASCII).");
        } else if (raw_mode) answer(line);
        else {
            const auto reply = dragon.respond(line);
            printf("Ember: %s\n", reply.text.c_str());
            display_begin_reply();
            display_append(reply.text.c_str());
            display_status("Ready / USB input");
            if (reply.memory_changed && !dragon_store.save(dragon.memory()))
                printf("[warning] %s\n", dragon_store.status().c_str());
        }
    }
    return true;
}

#ifdef ESP_PLATFORM
extern "C" void app_main() {
    if (esp_psram_get_size() < 8 * 1024 * 1024) {
        puts("[FATAL] Expected 8 MB PSRAM. Check N16R8 / Octal PSRAM settings.");
        return;
    }
    run_console(MODEL_DATA, MODEL_DATA_LEN, TOKENIZER_DATA, TOKENIZER_DATA_LEN, esp_random());
}
#else
static std::vector<uint8_t> load(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    const long size = ftell(f);
    if (size <= 0) { fclose(f); exit(1); }
    rewind(f);
    std::vector<uint8_t> data(size);
    if (fread(data.data(), 1, data.size(), f) != data.size()) { fclose(f); exit(1); }
    fclose(f);
    return data;
}
int main(int argc, char** argv) {
    if (argc != 3 && !(argc == 4 && std::strcmp(argv[3], "--raw") == 0)) {
        fprintf(stderr, "Usage: %s model.bin tokenizer.bin [--raw]\n", argv[0]); return 1;
    }
    raw_mode = argc == 4;
    const auto weights = load(argv[1]), vocab = load(argv[2]);
    return run_console(weights.data(), weights.size(), vocab.data(), vocab.size(), 1234) ? 0 : 1;
}
#endif
