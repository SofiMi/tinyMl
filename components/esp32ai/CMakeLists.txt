if(CMAKE_BUILD_EARLY_EXPANSION)
    idf_component_register(INCLUDE_DIRS "upstream")
    return()
endif()

set(generated "${CMAKE_CURRENT_BINARY_DIR}/generated")
execute_process(
    COMMAND "${PYTHON}" "${PROJECT_DIR}/tools/prepare_esp32ai.py" --out "${generated}"
    RESULT_VARIABLE prepare_result
)
if(NOT prepare_result EQUAL 0)
    message(FATAL_ERROR "Could not verify/generate esp32-ai assets")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${PROJECT_DIR}/tools/prepare_esp32ai.py"
    "${CMAKE_CURRENT_LIST_DIR}/upstream/tools/generate_tokenizer_header.py"
    "${CMAKE_CURRENT_LIST_DIR}/assets/model.bin"
    "${CMAKE_CURRENT_LIST_DIR}/assets/tokenizer.json"
    "${CMAKE_CURRENT_LIST_DIR}/assets/metadata.json")
idf_component_register(INCLUDE_DIRS "upstream" "${generated}")
# A normal `idf.py flash` must install the matching model as well as the app.
esptool_py_flash_to_partition(flash "model" "${CMAKE_CURRENT_LIST_DIR}/assets/model.bin")
