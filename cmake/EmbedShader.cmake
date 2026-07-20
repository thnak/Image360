# Converts a compiled SPIR-V (.spv) file into a C++ header defining a
# uint32_t array + byte size, so WindowsApp.Vulkan doesn't need to load
# shader binaries from disk at runtime (ctest's working directory isn't
# guaranteed relative to any install location).
#
# Invoked as:
#   cmake -DSPV_FILE=<path> -DHEADER_FILE=<path> -DARRAY_NAME=<identifier> -P EmbedShader.cmake
#
# Produces:
#   inline constexpr uint32_t <ARRAY_NAME>[] = { ... };
#   inline constexpr size_t <ARRAY_NAME>_size = sizeof(<ARRAY_NAME>);

if(NOT SPV_FILE OR NOT HEADER_FILE OR NOT ARRAY_NAME)
    message(FATAL_ERROR "EmbedShader.cmake requires SPV_FILE, HEADER_FILE, and ARRAY_NAME")
endif()

file(READ "${SPV_FILE}" hex_content HEX)
string(LENGTH "${hex_content}" hex_length)
math(EXPR word_count "${hex_length} / 8")

if(word_count LESS 1)
    message(FATAL_ERROR "EmbedShader.cmake: ${SPV_FILE} is empty or not word-aligned")
endif()

set(array_body "")
foreach(word_index RANGE 0 ${word_count})
    if(word_index EQUAL word_count)
        break()
    endif()

    math(EXPR char_offset "${word_index} * 8")
    string(SUBSTRING "${hex_content}" ${char_offset} 8 word_hex)

    # SPIR-V words are little-endian in the file - reverse byte order so
    # the literal parses to the same numeric value in C++.
    string(SUBSTRING "${word_hex}" 0 2 b0)
    string(SUBSTRING "${word_hex}" 2 2 b1)
    string(SUBSTRING "${word_hex}" 4 2 b2)
    string(SUBSTRING "${word_hex}" 6 2 b3)
    string(APPEND array_body "0x${b3}${b2}${b1}${b0},")
endforeach()

file(WRITE "${HEADER_FILE}"
"#pragma once
#include <cstdint>
#include <cstddef>
inline constexpr uint32_t ${ARRAY_NAME}[] = { ${array_body} };
inline constexpr size_t ${ARRAY_NAME}_size = sizeof(${ARRAY_NAME});
")
