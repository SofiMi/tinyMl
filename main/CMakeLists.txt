idf_component_register(
    SRCS "console.cpp" "display.cpp"
    INCLUDE_DIRS "."
    PRIV_REQUIRES esp32ai esp_partition esp_timer esp_psram esp_hw_support
                  esp_driver_i2c esp_driver_gpio esp_driver_spi
)

target_compile_options(${COMPONENT_LIB} PRIVATE -O3)
