idf_component_register(
    SRCS "console.cpp" "display.cpp"
    INCLUDE_DIRS "."
    PRIV_REQUIRES tinytalk esp_timer esp_psram esp_hw_support
                  esp_driver_i2c esp_driver_gpio esp_driver_spi
)
