#include "display.h"
#include "text_window.h"
#include "sdkconfig.h"

#if CONFIG_TINYML_OLED
#include "driver/i2c_master.h"
#include "esp_log.h"
#include "ssd1306.h"
#include <cstdio>

static const char* TAG = "tinyml_display";
static ssd1306_handle_t display;
static i2c_master_bus_handle_t bus;
static TextWindow text;
static char heading[22] = "Ember";

static void render() {
    if (!display) return;
    ssd1306_clear(display);
    ssd1306_draw_text(display, 0, 0, heading, true);
    for (int row = 0; row < 7; ++row)
        ssd1306_draw_text(display, 0, 8 + row * 8, text.line(row), true);
    const esp_err_t result = ssd1306_display(display);
    if (result != ESP_OK) {
        ESP_LOGW(TAG, "OLED error (%s); continuing on USB", esp_err_to_name(result));
        ssd1306_del(display);
        display = nullptr;
        i2c_del_master_bus(bus);
        bus = nullptr;
    }
}

void display_init() {
    i2c_master_bus_config_t bus_cfg = {};
    bus_cfg.i2c_port = I2C_NUM_0;
    bus_cfg.sda_io_num = static_cast<gpio_num_t>(CONFIG_TINYML_OLED_SDA);
    bus_cfg.scl_io_num = static_cast<gpio_num_t>(CONFIG_TINYML_OLED_SCL);
    bus_cfg.clk_source = I2C_CLK_SRC_DEFAULT;
    bus_cfg.glitch_ignore_cnt = 7;
    bus_cfg.flags.enable_internal_pullup = true;
    esp_err_t result = i2c_new_master_bus(&bus_cfg, &bus);
    if (result == ESP_OK) result = i2c_master_probe(bus, CONFIG_TINYML_OLED_ADDR, 50);
    if (result == ESP_OK) {
        ssd1306_config_t cfg = {};
        cfg.bus = SSD1306_I2C;
        cfg.width = 128;
        cfg.height = 64;
        cfg.iface.i2c.port = I2C_NUM_0;
        cfg.iface.i2c.addr = CONFIG_TINYML_OLED_ADDR;
        cfg.iface.i2c.rst_gpio = GPIO_NUM_NC;
        result = ssd1306_new_i2c(&cfg, &display);
    }
    if (result != ESP_OK) {
        ESP_LOGW(TAG, "OLED unavailable (%s); continuing on USB", esp_err_to_name(result));
        if (bus) { i2c_del_master_bus(bus); bus = nullptr; }
        return;
    }
    render();
}

void display_status(const char* status) {
    snprintf(heading, sizeof(heading), "%s", status);
    render();
}
void display_begin_reply() { text.clear(); display_status("Reading prompt..."); }
void display_append(const char* piece) { text.append(piece); render(); }
#else
void display_init() {}
void display_status(const char*) {}
void display_begin_reply() {}
void display_append(const char*) {}
#endif
