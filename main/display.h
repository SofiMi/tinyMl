#pragma once
#ifdef ESP_PLATFORM
void display_init();
void display_status(const char* status);
void display_begin_reply();
void display_append(const char* piece);
#else
inline void display_init() {}
inline void display_status(const char*) {}
inline void display_begin_reply() {}
inline void display_append(const char*) {}
#endif
