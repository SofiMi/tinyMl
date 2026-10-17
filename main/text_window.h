#pragma once
#include <array>

// Seven 21-character rows below the OLED's 8-pixel status line.
class TextWindow {
public:
    void clear() { rows_ = {}; row_ = col_ = 0; }
    void append(const char* text) {
        for (const unsigned char* p = reinterpret_cast<const unsigned char*>(text); *p; ++p) {
            if (*p == '\n') { newline(); continue; }
            if (*p < 32 || *p > 126) continue;
            if (col_ == 21) newline();
            rows_[row_][col_++] = static_cast<char>(*p);
            rows_[row_][col_] = '\0';
        }
    }
    const char* line(int index) const { return rows_[index].data(); }
private:
    void newline() {
        col_ = 0;
        if (row_ < 6) ++row_;
        else {
            for (int i = 0; i < 6; ++i) rows_[i] = rows_[i + 1];
            rows_[6] = {};
        }
    }
    std::array<std::array<char, 22>, 7> rows_{};
    int row_ = 0, col_ = 0;
};
