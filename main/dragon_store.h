#pragma once
#include "dragon.h"
#include <string>

// Load absence is normal; an error leaves the in-memory facts usable.
class DragonStore {
public:
    ~DragonStore();
    bool load(DragonMemory& memory);
    bool save(const DragonMemory& memory, bool reset_invalid = false);
    const std::string& status() const { return status_; }
private:
    bool ready_ = false;
    std::string status_ = "not initialized";
#ifdef ESP_PLATFORM
    uint32_t handle_ = 0;
#else
    std::string path_;
#endif
};
