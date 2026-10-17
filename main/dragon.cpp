#include "dragon.h"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
std::string trim(std::string s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    return s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}
std::string lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}
std::string sentence(std::string s) {
    s = trim(s);
    while (!s.empty() && (s.back() == '.' || s.back() == '!' || s.back() == '?')) s.pop_back();
    return trim(s);
}
bool starts(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }
bool has(const std::string& s, const char* part) { return s.find(part) != std::string::npos; }
bool one_of(const std::string& s, const char* words) {
    return s.find('|') == std::string::npos &&
        (std::string("|") + words + "|").find("|" + s + "|") != std::string::npos;
}
bool phrase(const std::string& s, size_t capacity, unsigned max_words = 3) {
    if (s.empty() || s.size() >= capacity || s.front() == ' ' || s.back() == ' ') return false;
    unsigned words = 1;
    bool letter = false;
    for (unsigned char c : s) {
        if (c == ' ') ++words;
        else if (c < 128 && std::isalpha(c)) letter = true;
        else if (c != '-' && c != '\'') return false;
    }
    return letter && words <= max_words && s.find("  ") == std::string::npos;
}
bool person(const std::string& s) {
    if (!phrase(s, 32, 2)) return false;
    const std::string q = lower(s);
    const std::string first = q.substr(0, q.find(' '));
    return !one_of(first, "not|a|an|the|your|my|you|i|hungry|sad|happy|scared|lonely|tired|fine|okay|ok|angry|cold|hot|hello|hi|hey|thanks|thank|yes|no|dragon|ember|human|sorry|ready|good|well|what|who|why|where|when|how|tell|play|bye|goodbye|help|story|count");
}
bool color(const std::string& s) { return one_of(s, "red|green|blue|yellow|orange|purple|pink|white|black|brown|grey|gray|gold|silver"); }
bool food(const std::string& s) { return one_of(s, "apple|apples|pear|pears|berries|carrots|bread|rice|pizza|nuts|peaches|chocolate|ice cream"); }
template<size_t N> void put(char (&field)[N], const std::string& value) {
    std::memset(field, 0, N);
    std::memcpy(field, value.data(), std::min(value.size(), N-1));
}
std::string value_or_unknown(const char* value) { return *value ? value : "unknown"; }
std::string strip_intro(std::string s) {
    for (int i = 0; i < 3; ++i) {
        bool changed = false;
        for (const char* p : {"hello, ", "hi, ", "no, ", "actually, ", "actually ", "sorry, ", "i made a mistake: "}) {
            if (starts(lower(s), p)) { s = trim(s.substr(std::strlen(p))); changed = true; break; }
        }
        if (!changed) break;
    }
    return s;
}
std::vector<std::string> clauses(const std::string& text) {
    std::vector<std::string> result;
    size_t pos = 0;
    while (pos < text.size()) {
        size_t end = text.find_first_of(".!;", pos);
        if (end == std::string::npos) end = text.size();
        auto part = trim(text.substr(pos, end-pos));
        if (!part.empty()) result.push_back(part);
        pos = end+1;
    }
    return result;
}
bool math(const std::string& input, long& a, long& b, char& op) {
    auto s = input;
    if (starts(s, "what is ")) s = trim(s.substr(8));
    else if (starts(s, "what's ")) s = trim(s.substr(7));
    const char* p = s.c_str();
    char* end = nullptr;
    a = std::strtol(p, &end, 10);
    if (end == p || a < -9999 || a > 9999) return false;
    while (*end == ' ') ++end;
    op = *end;
    if (op != '+' && op != '-') return false;
    p = end+1;
    b = std::strtol(p, &end, 10);
    if (end == p || b < -9999 || b > 9999) return false;
    while (*end == ' ') ++end;
    return !*end;
}
template<size_t N> bool valid_field(const char (&f)[N]) {
    const void* end = std::memchr(f, 0, N);
    if (!end) return false;
    return !*f || phrase(f, N);
}
}

bool valid_dragon_memory(const DragonMemory& m) {
    return m.version == 1 && valid_field(m.owner) && (!*m.owner || person(m.owner)) &&
        valid_field(m.friend_name) && (!*m.friend_name || person(m.friend_name)) &&
        valid_field(m.color) && (!*m.color || color(m.color)) && valid_field(m.food) && (!*m.food || food(m.food)) &&
        valid_field(m.home) && valid_field(m.user_item) && valid_field(m.dragon_item) &&
        valid_field(m.gift_color) && (!*m.gift_color || color(m.gift_color));
}
bool Dragon::restore(const DragonMemory& m) {
    if (!valid_dragon_memory(m)) return false;
    memory_ = m;
    new_conversation();
    return true;
}
void Dragon::new_conversation() {
    awaiting_name_ = counting_game_ = last_was_math_ = false;
    clarification_ = 0;
}
void Dragon::forget() { memory_ = DragonMemory{}; new_conversation(); }

std::string Dragon::describe_memory() const {
    return "Your name: " + value_or_unknown(memory_.owner) +
        "\nYour friend: " + value_or_unknown(memory_.friend_name) +
        "\nYour favorite color: " + value_or_unknown(memory_.color) +
        "\nYour favorite food: " + value_or_unknown(memory_.food) +
        "\nYour home: " + value_or_unknown(memory_.home) +
        "\nYour item: " + value_or_unknown(memory_.user_item) +
        "\nMy item: " + value_or_unknown(memory_.dragon_item) +
        "\nGift color: " + value_or_unknown(memory_.gift_color);
}

DragonReply Dragon::respond(const std::string& input) {
    if (input.size() > 256 || std::any_of(input.begin(), input.end(), [](unsigned char c) { return c >= 128; }))
        return {"Please use a short English message."};
    const std::string original = trim(input), q = lower(sentence(original));
    if (original.empty()) return {"I'm here. What would you like to do?"};
    const bool pending_name = awaiting_name_, pending_game = counting_game_;
    awaiting_name_ = counting_game_ = false;
    if (q.empty() || q == "huh" || q == "what") {
        ++clarification_;
        if (last_was_math_ && clarification_ == 1)
            return {"I counted " + std::to_string(math_a_) + " " + math_op_ + " " + std::to_string(math_b_) + ". That makes " + std::to_string(math_result_) + "."};
        if (clarification_ % 2) return {"I'm not sure what you mean. Try a full question, like 'What is my name?'"};
        return {"Would you like to count pebbles, hear a story, or tell me your name?"};
    }
    clarification_ = 0;
    last_was_math_ = false;
    long a, b; char op;
    if (math(q, a, b, op)) {
        math_a_ = a; math_b_ = b; math_op_ = op; math_result_ = op == '+' ? a+b : a-b;
        last_was_math_ = true;
        return {std::to_string(a) + " " + op + " " + std::to_string(b) + " = " + std::to_string(math_result_) + "."};
    }
    if (pending_game && q.size() <= 2 && std::all_of(q.begin(), q.end(), [](unsigned char c) { return std::isdigit(c); }))
        return {q == "5" ? "Five! You counted my pebbles. Well done!" : "Let's count together: three and two make five. Want to play again?"};

    const auto old = memory_;
    bool acknowledged = false;
    std::string name_ack;
    // Questions and quotations cannot introduce facts. Accept only explicit simple statements.
    if (original.find('?') == std::string::npos && original.find('"') == std::string::npos) {
        for (auto part : clauses(original)) {
            part = strip_intro(part);
            const std::string p = lower(part);
            for (const char* prefix : {"my name is ", "you can call me ", "call me ", "i am ", "i'm "}) {
                if (!starts(p, prefix)) continue;
                auto name = trim(part.substr(std::strlen(prefix)));
                const bool implicit = std::string(prefix) == "i am " || std::string(prefix) == "i'm ";
                if (person(name) && (!implicit || std::isupper(static_cast<unsigned char>(name.front())))) {
                    put(memory_.owner, name); name_ack = "Hello, " + name + "! I am Ember."; acknowledged = true;
                }
            }
            if (starts(p, "my friend is ")) {
                auto name = trim(part.substr(13));
                if (person(name)) { put(memory_.friend_name, name); acknowledged = true; }
            }
            for (const char* prefix : {"my favorite color is ", "my favourite colour is ", "i like "}) {
                if (!starts(p, prefix)) continue;
                auto v = trim(p.substr(std::strlen(prefix)));
                const auto negative = v.find(", not ");
                if (negative != std::string::npos) v = trim(v.substr(0, negative));
                if (color(v)) { put(memory_.color, v); acknowledged = true; }
                else if (std::string(prefix) == "i like " && food(v)) { put(memory_.food, v); acknowledged = true; }
            }
            for (const char* prefix : {"my favorite food is ", "my favourite food is "}) {
                if (starts(p, prefix)) {
                    auto v = trim(p.substr(std::strlen(prefix)));
                    if (food(v)) { put(memory_.food, v); acknowledged = true; }
                }
            }
            if (starts(p, "i live in ")) {
                auto v = trim(part.substr(10));
                if (phrase(v, sizeof(memory_.home))) { put(memory_.home, v); acknowledged = true; }
            }
            if (starts(p, "the ")) {
                for (const char* suffix : {" is mine", " is yours", " belongs to me", " belongs to you"}) {
                    const size_t size = std::strlen(suffix);
                    if (p.size() <= size+4 || p.compare(p.size()-size, size, suffix) != 0) continue;
                    const auto item = trim(p.substr(4, p.size()-size-4));
                    if (!phrase(item, sizeof(memory_.user_item))) continue;
                    const bool mine = std::string(suffix) == " is mine" || std::string(suffix) == " belongs to me";
                    if (mine) {
                        put(memory_.user_item, item);
                        if (item == memory_.dragon_item) { put(memory_.dragon_item, ""); put(memory_.gift_color, ""); }
                    } else {
                        put(memory_.dragon_item, item);
                        put(memory_.gift_color, "");
                        if (item == memory_.user_item) put(memory_.user_item, "");
                    }
                    acknowledged = true;
                }
            }
            for (const char* prefix : {"i gave you a ", "i give you a "}) {
                if (!starts(p, prefix)) continue;
                auto item = trim(p.substr(std::strlen(prefix)));
                if (!phrase(item, sizeof(memory_.dragon_item))) continue;
                const auto first = item.substr(0, item.find(' '));
                put(memory_.dragon_item, item);
                put(memory_.gift_color, color(first) ? first : "");
                if (item == memory_.user_item) put(memory_.user_item, "");
                acknowledged = true;
            }
        }
    }
    if (!acknowledged && pending_name && person(sentence(original)) &&
        sentence(original).find(' ') == std::string::npos && std::isupper(static_cast<unsigned char>(original.front())) &&
        original.find('?') == std::string::npos && original.find('"') == std::string::npos) {
        put(memory_.owner, sentence(original));
        name_ack = "Hello, " + std::string(memory_.owner) + "! I am Ember.";
        acknowledged = true;
    }
    if (acknowledged) {
        const bool changed = std::memcmp(&old, &memory_, sizeof(memory_)) != 0;
        if (!name_ack.empty()) return {name_ack, changed};
        std::string facts;
        if (std::strcmp(old.color, memory_.color)) facts += "You like " + std::string(memory_.color) + ". ";
        if (std::strcmp(old.food, memory_.food)) facts += "Your favorite food is " + std::string(memory_.food) + ". ";
        if (std::strcmp(old.home, memory_.home)) facts += "You live in " + std::string(memory_.home) + ". ";
        if (std::strcmp(old.friend_name, memory_.friend_name)) facts += "Your friend is " + std::string(memory_.friend_name) + ". ";
        if (*memory_.user_item && std::strcmp(old.user_item, memory_.user_item)) facts += "You have the " + std::string(memory_.user_item) + ". ";
        if (*memory_.dragon_item && std::strcmp(old.dragon_item, memory_.dragon_item)) facts += "I have the " + std::string(memory_.dragon_item) + ". ";
        return {facts.empty() ? "I remember. Thanks for telling me!" : trim(facts), changed};
    }

    if (q == "what are our names" || q == "what is your name and mine" ||
        (starts(q, "who is ") && has(q, "ember") && *memory_.owner && has(q, lower(memory_.owner).c_str())))
        return {*memory_.owner ? "I am Ember. You are " + std::string(memory_.owner) + "." : "I am Ember. I don't know your name yet."};
    if (has(q, "my friend's name")) return {*memory_.friend_name ? "Your friend is " + std::string(memory_.friend_name) + "." : "I don't know your friend's name yet."};
    if (has(q, "my name") || has(q, "corrected name") || q == "who am i" ||
        has(q, "what to call me") || has(q, "what should you call me") || has(q, "who is speaking to you")) {
        if (*memory_.owner) return {"Your name is " + std::string(memory_.owner) + ". I am Ember."};
        awaiting_name_ = true;
        return {"I don't know your name yet. What should I call you?"};
    }
    if (has(q, "your name") || q == "who are you" || q == "you are you" || q == "what are you" ||
        has(q, "about yourself") || has(q, "introduce yourself") || has(q, "what kind of creature"))
        return {"I am Ember, a friendly little dragon. I live in a warm cave and love apples."};
    if (has(q, "where do i live") || has(q, "what town do i live"))
        return {*memory_.home ? "You live in " + std::string(memory_.home) + "." : "I don't know where you live. You haven't told me yet."};
    if (has(q, "where do you live") || q == "where is your home") return {"My home is a warm cave."};
    if (has(q, "gift") && (has(q, "color") || has(q, "colour")))
        return {*memory_.gift_color ? "Your gift was " + std::string(memory_.gift_color) + "." : "I don't know the color of your gift yet."};
    if ((has(q, "color") || has(q, "colour")) && (has(q, "my ") || has(q, "do i like")))
        return {*memory_.color ? "You like " + std::string(memory_.color) + "." : "I don't know your favorite color. Try 'My favorite color is green'."};
    if (has(q, "my favorite food") || has(q, "my favourite food") || q == "what food do i like")
        return {*memory_.food ? "Your favorite food is " + std::string(memory_.food) + "." : "I don't know your favorite food yet."};
    if (has(q, "your favorite food") || has(q, "what do you eat") || has(q, "what food do you like")) return {"I love apples. Crunch!"};
    if (has(q, "belongs to me") || has(q, "is mine"))
        return {*memory_.user_item ? "The " + std::string(memory_.user_item) + " is yours." : "I don't know which item is yours yet."};
    if (has(q, "belongs to you") || has(q, "is yours"))
        return {*memory_.dragon_item ? "The " + std::string(memory_.dragon_item) + " is mine." : "You haven't given me an item yet."};
    if (one_of(q, "hello|hi|hey|hello ember|hi ember|good morning")) {
        if (*memory_.owner) return {"Hello, " + std::string(memory_.owner) + "! Want to count pebbles or hear a story?"};
        awaiting_name_ = true;
        return {"Hello! I am Ember, a little dragon. What should I call you?"};
    }
    if (has(q, "understand me") || has(q, "andestand me"))
        return {*memory_.owner ? "I follow short English messages. You are " + std::string(memory_.owner) + ", and I am Ember." : "I follow short English messages. Try telling me your name."};
    if (q == "i am hungry" || q == "i'm hungry") return {"My tummy rumbles too! I would pick a crunchy apple."};
    if (q == "i am sad" || q == "i feel sad" || q == "i feel lonely" || q == "i am scared" || has(q, "thunder"))
        return {"I will stay with you. We can sit quietly in my warm cave, or count shiny pebbles together."};
    if (has(q, "story")) {
        if (story_++ % 2 == 0) return {"Ember found a lost hat beside his cave. He carried it to a little girl. She smiled and gave him a shiny pebble."};
        return {"Ember planted an apple seed beside his cave. Each day he watered it. When a tiny leaf appeared, he invited his friends to celebrate."};
    }
    if (has(q, "play") || has(q, "game")) {
        counting_game_ = true;
        return {"Let's count pebbles! I have three and find two more. How many do I have?"};
    }
    if (q == "thanks" || q == "thank you") return {"You're welcome! A happy little roar for you."};
    if (q == "bye" || q == "goodbye") return {"See you soon! I'll keep our cave cozy."};
    if (starts(q, "you are ")) return {"I am Ember, your little dragon friend."};
    return {"I don't know how to answer that yet. We can talk about your name, colors, food or gifts, count pebbles, or play."};
}
