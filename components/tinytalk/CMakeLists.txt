if(CMAKE_BUILD_EARLY_EXPANSION)
    idf_component_register(INCLUDE_DIRS "include" PRIV_REQUIRES heap freertos)
    return()
endif()

set(MODEL_FILE "${CMAKE_CURRENT_LIST_DIR}/assets/model_neo_q4.bin")
set(TOKENIZER_FILE "${CMAKE_CURRENT_LIST_DIR}/assets/tok_neo.bin")
file(SHA256 "${MODEL_FILE}" model_hash)
file(SHA256 "${TOKENIZER_FILE}" tokenizer_hash)
if(NOT model_hash STREQUAL "183fe4c212fd0fd17a8e048ee3eab4a0095a039df290f0c58cafff70ca17fe2a"
   OR NOT tokenizer_hash STREQUAL "7b915b9779c12fa896ad839a81ce315c757419e71a4dee7f7890d2578051429f")
    message(FATAL_ERROR "TinyTalk model/tokenizer checksum mismatch. Restore the matching assets.")
endif()
configure_file(model_data.S.in "${CMAKE_CURRENT_BINARY_DIR}/model_data.S" @ONLY)
set_source_files_properties("${CMAKE_CURRENT_BINARY_DIR}/model_data.S"
    PROPERTIES OBJECT_DEPENDS "${MODEL_FILE};${TOKENIZER_FILE}")
idf_component_register(
    SRCS "llm.cpp" "dot_q4_pie.S" "${CMAKE_CURRENT_BINARY_DIR}/model_data.S"
    INCLUDE_DIRS "include"
    PRIV_REQUIRES heap freertos
)
