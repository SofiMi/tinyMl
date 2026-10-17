#if 0
#include <unistd.h>
#include <stdio.h>
#include "driver/i2c_master.h"
#include "esp_log.h"
#include "ssd1306.h"

static const char *TAG = "SSD1306_TEXT";

// --- Pin Definitions (match your wiring) ---
#define I2C_MASTER_SDA_IO   13
#define I2C_MASTER_SCL_IO   14
#define I2C_MASTER_FREQ_HZ  400000

// --- Display I2C Address (typically 0x3C or 0x3D) ---
#define SSD1306_I2C_ADDR    0x3C

void app_main(void)
{
    ESP_LOGI(TAG, "Starting SSD1306 text demo...");

    // 1. Initialize the I2C master bus
    i2c_master_bus_config_t bus_cfg = {
        .i2c_port = I2C_NUM_0,
        .sda_io_num = I2C_MASTER_SDA_IO,
        .scl_io_num = I2C_MASTER_SCL_IO,
        .clk_source = I2C_CLK_SRC_DEFAULT,
        .glitch_ignore_cnt = 7,
        .flags.enable_internal_pullup = true,
    };
    i2c_master_bus_handle_t bus_handle;
    ESP_ERROR_CHECK(i2c_new_master_bus(&bus_cfg, &bus_handle));

    // 2. Configure the SSD1306 driver
    ssd1306_config_t cfg = {
        .bus = SSD1306_I2C,
        .width = 128,
        .height = 64,
        .iface.i2c = {
            .port = I2C_NUM_0,
            .addr = SSD1306_I2C_ADDR,
            .rst_gpio = -1, // Set to a GPIO if your display has a reset pin
        },
    };

    // 3. Create the display handle
    ssd1306_handle_t disp;
    ESP_ERROR_CHECK(ssd1306_new_i2c(&cfg, &disp));

    // 4. Clear the screen
    ssd1306_clear(disp);

    // 5. Draw the text "Hello, world!" at coordinates (0, 0)
    ssd1306_draw_text(disp, 0, 0, "Hello, world!", true);

    // 6. Push the framebuffer to the display
    ssd1306_display(disp);

    ESP_LOGI(TAG, "Text drawn on display.");

    while (1)
    {
	    sleep(1);
    }
}
#else
#include <stdio.h>
#include <unistd.h>
#include "esp_log.h"
#include "esp_psram.h"        // For esp_psram_get_size()
#include "esp_heap_caps.h"    // For heap_caps_get_total_size()

static const char *TAG = "PSRAM_INFO";

void app_main(void)
{
    // Ensure PSRAM is initialized before calling these functions
    // (It is initialized automatically during system startup if enabled in menuconfig)
    
    // --- Method 1: esp_psram_get_size() ---
    size_t psram_size = esp_psram_get_size();
    if (psram_size > 0) {
        ESP_LOGI(TAG, "esp_psram_get_size() -> %u bytes (%.2f MB)", 
                 psram_size, psram_size / (1024.0 * 1024.0));
    } else {
        ESP_LOGW(TAG, "esp_psram_get_size() returned 0. PSRAM may not be enabled or present.");
    }

    // --- Method 2: heap_caps_get_total_size() ---
    size_t total_psram = heap_caps_get_total_size(MALLOC_CAP_SPIRAM);
    size_t free_psram  = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    ESP_LOGI(TAG, "heap_caps_get_total_size(MALLOC_CAP_SPIRAM) -> %u bytes", total_psram);
    ESP_LOGI(TAG, "heap_caps_get_free_size(MALLOC_CAP_SPIRAM)  -> %u bytes", free_psram);

    // Idle loop
    while (1) {
	printf("Hello world");
        sleep(1);
    }
}
#endif
