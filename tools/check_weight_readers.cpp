// Compare Python audit readers with the firmware's native dequantizers.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>
#ifdef CHECK_PLE
#include "components/esp32ai/upstream/llm.h"
static void emit(const std::string& name, const QT& t) {
    std::vector<float> values(t.cols);
    for (int r : {0, t.rows / 2, t.rows - 1}) {
        deq_row(&t, r, values.data());
        for (int c : {0, t.cols / 2, t.cols - 1})
            std::printf("%s,%d,%d,%a\n", name.c_str(), r, c, values[c]);
    }
}
#else
#include "components/tinytalk/llm.cpp"
static void emit(const std::string& name, const uint8_t* data, int rows, int cols) {
    std::vector<float> values(cols);
    for (int r : {0, rows / 2, rows - 1}) {
        dequant_q4_row_v3(values.data(), data + r * v3_row_stride(cols), cols);
        for (int c : {0, cols / 2, cols - 1})
            std::printf("%s,%d,%d,%a\n", name.c_str(), r, c, values[c]);
    }
}
#endif
int main(int argc, char** argv) {
    if (argc != 2) return 2;
    std::ifstream file(argv[1], std::ios::binary);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)), {});
    if (data.empty()) return 3;
#ifdef CHECK_PLE
    Model m{};
    if (llm_load(data.data(), &m) || m.image_bytes != data.size()) return 4;
    emit("tok_emb", m.tok_emb);
    emit("ple_model_proj", m.ple_model_proj);
    emit("ple_table", m.ple_table);
    for (int i = 0; i < m.c.n_layers; ++i) {
        std::string p = "blk." + std::to_string(i) + ".";
        emit(p + "qkv", m.qkv[i]); emit(p + "attn_proj", m.attn_proj[i]);
        emit(p + "gate", m.gate[i]); emit(p + "up", m.up[i]);
        emit(p + "down", m.down[i]); emit(p + "ple_gate", m.ple_gate[i]);
        emit(p + "ple_proj", m.ple_proj[i]);
    }
#else
    Transformer t{};
    if (!llm_init_embedded(&t, data.data(), data.size(), 1)) return 4;
    const Config& c = t.config;
    emit("wte", t.token_embed_q4, c.vocab_size, c.dim);
    for (int i = 0; i < c.n_layers; ++i) {
        std::string p = "blk." + std::to_string(i) + ".";
        emit(p + "wq", t.wq_q4 + i*t.stride_wq, c.dim, c.dim);
        emit(p + "wk", t.wk_q4 + i*t.stride_wk, c.dim, c.dim);
        emit(p + "wv", t.wv_q4 + i*t.stride_wv, c.dim, c.dim);
        emit(p + "wo", t.wo_q4 + i*t.stride_wo, c.dim, c.dim);
        emit(p + "w_fc", t.w1_q4 + i*t.stride_w1w3, c.hidden_dim, c.dim);
        emit(p + "w_proj", t.w2_q4 + i*t.stride_w2, c.dim, c.hidden_dim);
    }
#endif
}
