#pragma once
#ifdef ESP_PLATFORM
void display_init();
bool display_is_present();
void display_status(const char* status);
void display_begin_reply();
void display_append(const char* piece);
#else
inline void display_init() {}
inline bool display_is_present() { return false; }
inline void display_status(const char*) {}
inline void display_begin_reply() {}
inline void display_append(const char*) {}
#endif
