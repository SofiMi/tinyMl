#include "dragon_store.h"
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>

#ifdef ESP_PLATFORM
#include "nvs.h"
#include "nvs_flash.h"
#endif

DragonStore::~DragonStore() {
#ifdef ESP_PLATFORM
    if (ready_) nvs_close(handle_);
#endif
}

bool DragonStore::load(DragonMemory& memory) {
    DragonMemory candidate;
#ifdef ESP_PLATFORM
    esp_err_t err = nvs_flash_init();
    // Do not erase NVS on errors: other applications/settings may use it.
    if (err == ESP_OK) err = nvs_open("tinyml_dragon", NVS_READWRITE, &handle_);
    if (err != ESP_OK) { status_ = std::string("volatile: ") + esp_err_to_name(err); return false; }
    ready_ = true;
    size_t size = sizeof(candidate);
    err = nvs_get_blob(handle_, "memory_v1", &candidate, &size);
    if (err == ESP_ERR_NVS_NOT_FOUND) { status_ = "NVS (no saved facts yet)"; return true; }
    if (err != ESP_OK || size != sizeof(candidate) || !valid_dragon_memory(candidate)) {
        status_ = "invalid saved facts; use /forget to reset";
        ready_ = false;
        nvs_close(handle_);
        return false;
    }
    status_ = "NVS (loaded)";
#else
    const char* configured = std::getenv("TINYML_DRAGON_MEMORY");
    if (!configured || !*configured) { status_ = "volatile (no memory file configured)"; return true; }
    path_ = configured;
    ready_ = true;
    FILE* file = std::fopen(path_.c_str(), "rb");
    if (!file) {
        if (errno == ENOENT) { status_ = "file (no saved facts yet)"; return true; }
        status_ = std::string("memory file read failed: ") + std::strerror(errno);
        ready_ = false;
        return false;
    }
    const bool complete = std::fread(&candidate, 1, sizeof(candidate), file) == sizeof(candidate);
    const int trailing = std::fgetc(file);
    const bool read_error = std::ferror(file);
    std::fclose(file);
    if (!complete || trailing != EOF || read_error || !valid_dragon_memory(candidate)) {
        status_ = "invalid saved facts; use /forget to reset";
        ready_ = false;
        return false;
    }
    status_ = "file (loaded)";
#endif
    memory = candidate;
    return true;
}

bool DragonStore::save(const DragonMemory& memory, bool reset_invalid) {
    if (!valid_dragon_memory(memory)) { status_ = "invalid memory record"; return false; }
#ifdef ESP_PLATFORM
    if (!ready_ && reset_invalid && nvs_flash_init() == ESP_OK)
        ready_ = nvs_open("tinyml_dragon", NVS_READWRITE, &handle_) == ESP_OK;
    if (!ready_) { status_ = "NVS unavailable; facts are volatile"; return false; }
    esp_err_t err = nvs_set_blob(handle_, "memory_v1", &memory, sizeof(memory));
    if (err == ESP_OK) err = nvs_commit(handle_);
    if (err != ESP_OK) { status_ = std::string("NVS save failed: ") + esp_err_to_name(err); return false; }
    status_ = "NVS (saved)";
#else
    if (path_.empty()) { status_ = "volatile (no memory file configured)"; return true; }
    if (reset_invalid) ready_ = true;
    if (!ready_) { status_ = "memory file unavailable; facts are volatile"; return false; }
    const std::string temporary = path_ + ".tmp";
    FILE* file = std::fopen(temporary.c_str(), "wb");
    if (!file) { status_ = "cannot create memory file; facts are volatile"; return false; }
    bool ok = std::fwrite(&memory, 1, sizeof(memory), file) == sizeof(memory);
    if (std::fclose(file) != 0) ok = false;
    if (!ok || std::rename(temporary.c_str(), path_.c_str()) != 0) {
        std::remove(temporary.c_str());
        status_ = "memory file save failed; facts are volatile";
        return false;
    }
    status_ = "file (saved)";
#endif
    return true;
}
