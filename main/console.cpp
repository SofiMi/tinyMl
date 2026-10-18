// PLE TinyStories USB-UART console. Runtime and memory placement from slvDev/esp32-ai.
#include "display.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef ESP_PLATFORM
#include "esp_heap_caps.h"
#include "esp_psram.h"
#include "esp_partition.h"
#include "esp_random.h"
#include "esp_timer.h"
#include "esp_stdio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#endif
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
    vTaskDelay(1);
#endif
}

#if defined(ESP_PLATFORM) && CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG && !CONFIG_VFS_SUPPORT_IO
#error "Interactive USB console requires CONFIG_VFS_SUPPORT_IO=y"
#endif

#define LLM_PROFILE 1
#define LLM_PROFILE_NOW() micros()
#define LLM_INT8_ACT 1
#include "llm.h"
#include "bpe_tokenizer.h"
#include "tokenizer_encoder.h"
#include "vocab.h"

static Model model;
static Scratch s;
static BpeTokenizer tokenizer;
static constexpr size_t kModelBytes = 14912348;
static constexpr size_t kMaxInput = 256;
static constexpr int kMaxReply = 128;
static float temperature = 0.0f;
static uint32_t random_state = 1234;
static bool dual_core_active = false;

// ---- allocation ------------------------------------------------------------
// Allocations are strict because memory placement is part of the runtime
// configuration. Allocation failure stops initialization.
static size_t psram_used = 0, sram_used = 0;

// The two LLM_Q8_MAX_INPUT int8 activation buffers (matvec_q8, matvec_par),
// which are static and so absent from the totals above. Together these report
// the managed hot set, not total SRAM usage; the free-SRAM figure printed at
// boot is the overall diagnostic.
#ifdef ESP_PLATFORM
#define STATIC_SRAM_BYTES (2 * LLM_Q8_MAX_INPUT)
#else
#define STATIC_SRAM_BYTES LLM_Q8_MAX_INPUT
#endif

static void *ps(size_t n) {
  #ifdef ESP_PLATFORM
  void *p = heap_caps_malloc(n, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
#else
  void *p = malloc(n);
#endif
  if (p) psram_used += n;
  return p;
}
static void *ps_or_die(size_t n, const char *what) {
  void *p = ps(n);
  if (!p) {
    printf("FATAL: required PSRAM allocation failed: %s (%u bytes)\n",
                  what, (unsigned)n);
    abort();
  }
  return p;
}
static void *sram_or_die(size_t n, const char *what) {
  #ifdef ESP_PLATFORM
  void *p = heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
#else
  void *p = malloc(n);
#endif
  if (!p) {
    printf("FATAL: required SRAM allocation failed: %s (%u bytes)\n",
                  what, (unsigned)n);
    abort();
  }
  sram_used += n;
  return p;
}

// ---- dual-core int8 matvec -------------------------------------------------
// Serves both hooks. Below ~128 rows the task notify round trip costs more than
// the split saves, so small tensors run single-core.
#ifdef ESP_PLATFORM
static TaskHandle_t worker_h, main_h;
static const QT *job_t;
static const int8_t *job_xq;
static float job_xs;
static float *job_y;
static int job_split;

static void worker_main(void *) {
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    matvec_i8_range(job_t, job_xq, job_xs, job_y, 0, job_split);
    xTaskNotifyGive(main_h);
  }
}

static void matvec_par(const QT *t, const float *x, float *y) {
  static int8_t xq[LLM_Q8_MAX_INPUT];
  float xs;
  if (t->w8 == NULL || t->rows < 128) { MATVEC(t, x, y); return; }
  quantize_act(x, t->cols, xq, &xs);   // once; both cores read the result
  job_t = t; job_xq = xq; job_xs = xs; job_y = y; job_split = t->rows / 2;
  xTaskNotifyGive(worker_h);
  matvec_i8_range(t, xq, xs, y, job_split, t->rows);
  ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
}

#endif

// Copy RMSNorm weights from mapped flash to internal SRAM.
static void copy_norms_to_sram() {
  Cfg *c = &model.c;
  int D = c->dim, L = c->n_layers, P = c->ple_dim;
  const float **vecs[3 * 32 + 2];
  int sizes[3 * 32 + 2], n_vec = 0;
  vecs[n_vec] = &model.ple_proj_norm; sizes[n_vec++] = P;
  for (int l = 0; l < L; l++) {
    vecs[n_vec] = &model.attn_norm[l]; sizes[n_vec++] = D;
    vecs[n_vec] = &model.ffn_norm[l];  sizes[n_vec++] = D;
    vecs[n_vec] = &model.ple_norm[l];  sizes[n_vec++] = D;
  }
  vecs[n_vec] = &model.out_norm; sizes[n_vec++] = D;
  for (int i = 0; i < n_vec; i++) {
    size_t bytes = (size_t)sizes[i] * sizeof(float);
    void *dst = sram_or_die(bytes, "norm vector");
    memcpy(dst, *vecs[i], bytes);
    *vecs[i] = (const float *)dst;
  }
  printf("norms  -> SRAM   %d vectors\n", n_vec);
}

static void alloc_scratch() {
  Cfg *c = &model.c;
  int D = c->dim, L = c->n_layers, P = c->ple_dim, F = c->ffn, S = c->seq_len;
  // hot working set -> internal SRAM
  s.x     = (float *)sram_or_die(D * 4, "x");
  s.h     = (float *)sram_or_die((F > D ? F : D) * 4, "h");
  s.qkv   = (float *)sram_or_die(3 * D * 4, "qkv");
  s.att   = (float *)sram_or_die(D * 4, "att");
  s.g1    = (float *)sram_or_die(F * 4, "g1");
  s.g2    = (float *)sram_or_die((P > F ? P : F) * 4, "g2");
  s.ple   = (float *)sram_or_die(L * P * 4, "ple");
  s.tmpP  = (float *)sram_or_die(L * P * 4, "tmpP");
  s.trow  = (float *)sram_or_die(L * P * 4, "trow");
  s.scores = (float *)sram_or_die(S * 4, "scores");
  // logits: out_vocab floats, 99 KiB here, read once per token. Left in PSRAM
  // rather than spend a fifth of internal SRAM on it.
  s.logits = (float *)ps_or_die((size_t)model.out_vocab * 4, "logits");
  // KV cache: 1.1MB, read once per position rather than per matvec.
  s.kcache = (float *)ps_or_die((size_t)L * S * D * 4, "kcache");
  s.vcache = (float *)ps_or_die((size_t)L * S * D * 4, "vcache");
}


static void stats() {
    printf("[model] PLE TinyStories 28.9M, vocab=%d, KV=%d, temperature=%.2f, history=0\n",
           model.out_vocab, model.c.seq_len, temperature);
    printf("[allocation] dynamic SRAM=%zu, static SRAM=%d, PSRAM=%zu bytes\n",
           sram_used, STATIC_SRAM_BYTES, psram_used);
    printf("[config] int8=1 dual_core=%d profile=1 display_present=%d\n",
           dual_core_active, display_is_present());
#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
    puts("[console] input=USB_SERIAL_JTAG output=USB_SERIAL_JTAG");
#elif CONFIG_ESP_CONSOLE_SECONDARY_USB_SERIAL_JTAG
    puts("[console] input=UART0 output=UART0+USB_SERIAL_JTAG");
#endif
#ifdef ESP_PLATFORM
    printf("[config] CPU=%dMHz PSRAM=%dMHz cache_line=%d UART=%d baud=%d OLED=%d\n",
           CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ, CONFIG_SPIRAM_SPEED,
           CONFIG_ESP32S3_DATA_CACHE_LINE_SIZE, CONFIG_ESP_CONSOLE_UART_NUM,
#ifdef CONFIG_ESP_CONSOLE_UART_BAUDRATE
           CONFIG_ESP_CONSOLE_UART_BAUDRATE,
#else
           0,
#endif
#ifdef CONFIG_TINYML_OLED
           1);
#else
           0);
#endif
#endif
#ifdef ESP_PLATFORM
    printf("[memory] PSRAM=%zu, free PSRAM=%zu, free internal=%zu bytes\n",
           esp_psram_get_size(), heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
           heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
#endif
}

static int sample() {
    int best = 0;
    for (int i = 1; i < model.out_vocab; ++i)
        if (s.logits[i] > s.logits[best]) best = i;
    if (temperature == 0) return best;
    const float max_logit = s.logits[best];
    double sum = 0;
    for (int i = 0; i < model.out_vocab; ++i) {
        s.logits[i] = expf((s.logits[i] - max_logit) / temperature);
        sum += s.logits[i];
    }
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    double threshold = (random_state / 4294967296.0) * sum;
    for (int i = 0; i < model.out_vocab; ++i) {
        threshold -= s.logits[i];
        if (threshold < 0) return i;
    }
    return best;
}

static void answer(const std::string& prompt, bool benchmark = false, bool oled = true) {
    const int limit = benchmark ? 200 : kMaxReply;
    // The BPE runtime supports ordinary ASCII, not literal added special tokens.
    if (prompt.find("<|endoftext|>") != std::string::npos) {
        puts("[error] The EOS marker is reserved; enter ordinary story text.");
        return;
    }
    uint16_t tokens[kMaxInput];
    int count = bpe_encode_ascii(&tokenizer, prompt.c_str(), tokens, kMaxInput);
    if (count <= 0 || count > model.c.seq_len - limit) {
        printf("[error] Prompt exceeds the %d-token budget or cannot be encoded.\n",
               model.c.seq_len - limit);
        return;
    }
    // Every line starts a fresh story. Position zero overwrites the old KV;
    // attention only reads positions already written by this request.
    printf("[reading %d tokens]\n", count);
    if (oled) {
        display_begin_reply();
        display_status("Thinking...");
    }
    if (benchmark) printf("[benchmark] prompt=Once upon a time, tokens=200, greedy=1, oled=%d display_present=%d\n",
                          oled, display_is_present());
    fputs("Story:", stdout);
    const int64_t started = micros();
    int pos = 0;
    for (int i = 0; i < count; ++i) {
        llm_forward(&model, tokens[i], pos++, &s);
        if ((i & 7) == 7) yield_cpu();
    }
    llm_profile_reset(&s);
    int64_t compute_us = 0;
    const int64_t generation_started = micros();
    int outputs = 0;
    const char* stop = "reply limit";
    for (int i = 0; i < limit; ++i) {
        const float previous_temp = temperature;
        if (benchmark) temperature = 0;
        const int token = sample();
        temperature = previous_temp;
        if (!benchmark && token == EOS_ID) { stop = "EOS"; break; }
        const unsigned char* bytes = VOCAB_BLOB + VOCAB_OFF[token];
        const size_t len = VOCAB_OFF[token + 1] - VOCAB_OFF[token];
        // Suppress terminal control bytes (except newline/tab) in generated text.
        std::string piece;
        for (size_t j = 0; j < len; ++j)
            if (bytes[j] >= 32 && bytes[j] != 127) piece += static_cast<char>(bytes[j]);
            else if (bytes[j] == '\n' || bytes[j] == '\t') piece += static_cast<char>(bytes[j]);
        fputs(piece.c_str(), stdout);
        if (oled) display_append(piece.c_str());
        ++outputs;
        if (!benchmark && i + 1 == limit) break;
        if (pos >= model.c.seq_len) { stop = "context limit"; break; }
        const int64_t step_start = micros();
        llm_forward(&model, token, pos++, &s);
        compute_us += micros() - step_start;
        if ((i & 7) == 7) yield_cpu();
    }
    // Flush the final OLED frame inside the measured end-to-end interval.
    if (oled) display_status("Ready / USB input");
    const double seconds = (micros() - generation_started) / 1000000.0;
    printf("\n[done] %d tokens, read=%.2fs, generation=%.2fs, %.2f tokens/s, stop=%s\n",
           outputs, (generation_started - started) / 1000000.0, seconds,
           seconds > 0 ? outputs / seconds : 0.0, stop);
    if (s.profile.calls) {
        const double n = s.profile.calls * 1000.0;
        printf("[timing] forwards=%u compute=%.2f ms/token end_to_end=%.2f tokens/s\n",
               static_cast<unsigned>(s.profile.calls), compute_us / n,
               seconds > 0 ? outputs / seconds : 0.0);
        printf("[profile ms/token] input=%.2f attention=%.2f ffn=%.2f ple=%.2f head=%.2f\n",
               s.profile.input_us / n, s.profile.attn_us / n, s.profile.ffn_us / n,
               s.profile.ple_us / n, s.profile.head_us / n);
    }
}

static void help() {
    puts("Type an English story opening and press Enter (e.g. Once upon a time).\n"
         "Commands: /new, /stats, /temp 0..2, /bench, /bench oled, /help\n"
         "TinyStories continues text; it is not trained for chat or instructions.\n"
         "Each line starts a new story; output is limited to 128 tokens.");
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


static bool run_console(const uint8_t* weights, size_t weights_size) {
    setvbuf(stdin, nullptr, _IONBF, 0);
    setvbuf(stdout, nullptr, _IONBF, 0);
    puts("\nPLE TinyStories 28.9M / ESP32-S3 N16R8 / USB console");
    // Check the known image before calling upstream's pointer-only loader.
    if (weights_size < kModelBytes) {
        puts("[FATAL] Model partition/file is too small.");
        return false;
    }
    uint32_t fp = 2166136261u;
    for (size_t i = 0; i < kModelBytes; ++i) {
        fp ^= weights[i]; fp *= 16777619u;
        if ((i & 0x3ffff) == 0) yield_cpu();
    }
    if (fp != 0xa9bdd778u) {
        puts("[FATAL] Model fingerprint mismatch; run idf.py flash (including model).");
        return false;
    }
    if (llm_load(weights, &model) || model.image_bytes != kModelBytes ||
        model.out_vocab != VOCAB_N ||
        bpe_tokenizer_load(TOKENIZER_ENCODER_ASSET, TOKENIZER_ENCODER_ASSET_SIZE, &tokenizer) ||
        tokenizer.active_vocab != VOCAB_N || tokenizer.active_vocab > (unsigned)model.c.vocab) {
        puts("[FATAL] Model/tokenizer initialization failed.");
        return false;
    }
    alloc_scratch();
    copy_norms_to_sram();
    const int want = llm_core_stage_count(&model);
    if (llm_stage_core_int8_alloc(&model, ps) != want) {
        puts("[FATAL] Could not stage core tensors in PSRAM.");
        return false;
    }
    llm_stage_int8(&model.out_head, ps_or_die(llm_stage_int8_bytes(&model.out_head), "head"));
#ifdef ESP_PLATFORM
    main_h = xTaskGetCurrentTaskHandle();
    const int worker_core = xPortGetCoreID() == 0 ? 1 : 0;
    if (xTaskCreatePinnedToCore(worker_main, "llm-matvec", 4096, nullptr, 2,
                                &worker_h, worker_core) == pdPASS) {
        model.layer_matvec = matvec_par;
        model.head_matvec = matvec_par;
        dual_core_active = true;
    } else puts("[warning] Dual-core worker failed; using single core.");
    random_state = esp_random();
    if (!random_state) random_state = 1234;
#endif
    printf("[build] model_bytes=%zu fp=%08x staged=%d\n", model.image_bytes,
           static_cast<unsigned>(fp), want + 1);
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
        if (line == "/new") {
            display_begin_reply();
            display_status("New story");
            puts("[new story; each input starts with empty context]");
        } else if (line == "/stats") stats();
        else if (line == "/help") help();
        else if (line == "/bench") answer("Once upon a time", true, false);
        else if (line == "/bench oled") answer("Once upon a time", true, true);
        else if (line.compare(0, 6, "/temp ") == 0) {
            char* end = nullptr;
            const float value = strtof(line.c_str() + 6, &end);
            if (end == line.c_str() + 6 || *end || !std::isfinite(value) || value < 0 || value > 2)
                puts("[error] Use /temp followed by a number from 0 to 2.");
            else { temperature = value; stats(); }
        } else if (line[0] == '/') puts("[error] Unknown command. Use /help.");
        else if (std::any_of(line.begin(), line.end(), [](unsigned char c) { return c >= 128; }))
            puts("[error] This encoder supports English ASCII input only.");
        else answer(line);
    }
    return true;
}

#ifdef ESP_PLATFORM
extern "C" void app_main() {
#if CONFIG_ESP_CONSOLE_USB_SERIAL_JTAG
    // Interrupt-driven USB I/O flushes short prompts without waiting for a
    // newline/full hardware FIFO, and lets getchar() sleep while awaiting input.
    const esp_err_t io_error = esp_stdio_install_io_driver();
    if (io_error != ESP_OK) {
        printf("[FATAL] USB console driver: %s\n", esp_err_to_name(io_error));
        return;
    }
#endif
    if (esp_psram_get_size() < 8 * 1024 * 1024) {
        puts("[FATAL] Expected 8 MB PSRAM. Check N16R8 / Octal PSRAM settings.");
        return;
    }
    const esp_partition_t* part = esp_partition_find_first(
        ESP_PARTITION_TYPE_DATA, static_cast<esp_partition_subtype_t>(0x40), "model");
    if (!part || part->size < kModelBytes) {
        puts("[FATAL] Missing/undersized model partition. Run idf.py flash.");
        return;
    }
    const void* base = nullptr;
    esp_partition_mmap_handle_t handle;
    if (esp_partition_mmap(part, 0, kModelBytes, ESP_PARTITION_MMAP_DATA, &base, &handle) != ESP_OK) {
        puts("[FATAL] Could not map the model partition.");
        return;
    }
    run_console(static_cast<const uint8_t*>(base), kModelBytes);
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
    if (argc != 2) { fprintf(stderr, "Usage: %s model.bin\n", argv[0]); return 1; }
    const auto weights = load(argv[1]);
    return run_console(weights.data(), weights.size()) ? 0 : 1;
}
#endif
