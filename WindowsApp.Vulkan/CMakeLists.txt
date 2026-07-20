# WindowsApp.Vulkan: Vulkan compute backend (GLSL/SPIR-V kernels for
# WarpPerspective/MedianStack/ApplyGain/DemosaicBayer; the remaining
# IComputeBackend methods delegate to WindowsApp.Core's portable CPU
# kernels - see VulkanPipeline.h). Only added to the build when the
# top-level CMakeLists.txt found a Vulkan SDK and glslc - see
# IMAGE360_HAVE_VULKAN there.

set(VULKAN_SHADER_NAMES
    apply_gain
    warp_perspective
    median_stack
    demosaic_bayer
)

set(GENERATED_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_SHADER_DIR})

set(GENERATED_SHADER_HEADERS "")
foreach(shader_name ${VULKAN_SHADER_NAMES})
    set(comp_file ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader_name}.comp)
    set(spv_file ${GENERATED_SHADER_DIR}/${shader_name}.spv)
    set(header_file ${GENERATED_SHADER_DIR}/${shader_name}.spv.h)

    add_custom_command(
        OUTPUT ${spv_file}
        COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.2 -O ${comp_file} -o ${spv_file}
        DEPENDS ${comp_file}
        COMMENT "Compiling ${shader_name}.comp -> SPIR-V"
        VERBATIM
    )

    add_custom_command(
        OUTPUT ${header_file}
        COMMAND ${CMAKE_COMMAND}
            -DSPV_FILE=${spv_file}
            -DHEADER_FILE=${header_file}
            -DARRAY_NAME=${shader_name}_spv
            -P ${CMAKE_SOURCE_DIR}/cmake/EmbedShader.cmake
        DEPENDS ${spv_file} ${CMAKE_SOURCE_DIR}/cmake/EmbedShader.cmake
        COMMENT "Embedding ${shader_name}.spv -> C++ header"
        VERBATIM
    )

    list(APPEND GENERATED_SHADER_HEADERS ${header_file})
endforeach()

add_custom_target(windowsapp_vulkan_shaders DEPENDS ${GENERATED_SHADER_HEADERS})

add_library(windowsapp_vulkan STATIC
    SourceFiles/VulkanPipeline.cpp
)
add_dependencies(windowsapp_vulkan windowsapp_vulkan_shaders)

target_include_directories(windowsapp_vulkan
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/HeaderFiles
    PRIVATE
        ${GENERATED_SHADER_DIR}
)

target_link_libraries(windowsapp_vulkan
    PUBLIC
        Vulkan::Vulkan
    PRIVATE
        windowsapp_core
)
