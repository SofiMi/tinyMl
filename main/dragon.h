#pragma once
#include <cstdint>
#include <string>

// Versioned, bounded facts. Generated text is never an input to this record.
struct DragonMemory {
    uint32_t version = 1;
    char owner[32] = {};
    char friend_name[32] = {};
    char color[16] = {};
    char food[24] = {};
    char home[32] = {};
    char user_item[24] = {};
    char dragon_item[24] = {};
    char gift_color[16] = {};
};
static_assert(sizeof(DragonMemory) == 204, "Update persistent format version if layout changes");
bool valid_dragon_memory(const DragonMemory& memory);

struct DragonReply {
    std::string text;
    bool memory_changed = false;
};

class Dragon {
public:
    const DragonMemory& memory() const { return memory_; }
    bool restore(const DragonMemory& memory);
    void new_conversation();
    void forget();
    DragonReply respond(const std::string& input);
    std::string describe_memory() const;
private:
    DragonMemory memory_;
    bool awaiting_name_ = false;
    bool counting_game_ = false;
    unsigned clarification_ = 0;
    unsigned story_ = 0;
    bool last_was_math_ = false;
    long math_a_ = 0, math_b_ = 0, math_result_ = 0;
    char math_op_ = '+';
};
