# WindowsApp.Core: orchestration layer (RAW decode via vendored LibRaw,
# SQLite project DB, task executors, storage engine). Builds standalone
# without CUDA - Compute-dependent code paths (AlignExecutor etc.) just
# won't link into anything that doesn't reference them.

set(CORE_SOURCES
    SourceFiles/AlignExecutor.cpp
    SourceFiles/BriefDescriptorExtractor.cpp
    SourceFiles/BundleAdjustment.cpp
    SourceFiles/CpuComputeBackend.cpp
    SourceFiles/CpuSimdDetect.cpp
    SourceFiles/FastFeatureDetector.cpp
    SourceFiles/FeatureMatcher.cpp
    SourceFiles/GainColorOps.cpp
    SourceFiles/HomographyMath.cpp
    SourceFiles/ImageLoader.cpp
    SourceFiles/JpegCodec.cpp
    SourceFiles/LinearSolve.cpp
    SourceFiles/OptimizeExecutor.cpp
    SourceFiles/OverlapCulling.cpp
    SourceFiles/PanoramaExporter.cpp
    SourceFiles/PipelineDriver.cpp
    SourceFiles/PlatformFile.cpp
    SourceFiles/ProjectManager.cpp
    SourceFiles/RansacHomography.cpp
    SourceFiles/RawIngestExecutor.cpp
    SourceFiles/RenderExecutor.cpp
    SourceFiles/RgbToGray.cpp
    SourceFiles/StorageEngine.cpp
    SourceFiles/TaskScheduler.cpp
    SourceFiles/TextEncoding.cpp
)

# Tiered SIMD kernels (MedianStack/WarpPerspective/BayerDemosaic) - each
# compiled as a separate translation unit per tier since MSVC has no
# GCC-style function multiversioning; CpuComputeBackend picks which tier's
# function pointers to call at runtime via CpuSimdDetect.
set(TIERED_KERNEL_SCALAR_SOURCES
    SourceFiles/BayerDemosaic_Scalar.cpp
    SourceFiles/MedianStack_Scalar.cpp
    SourceFiles/WarpPerspective_Scalar.cpp
)
set(TIERED_KERNEL_AVX2_SOURCES
    SourceFiles/BayerDemosaic_Avx2.cpp
    SourceFiles/MedianStack_Avx2.cpp
    SourceFiles/WarpPerspective_Avx2.cpp
)
set(TIERED_KERNEL_AVX512_SOURCES
    SourceFiles/BayerDemosaic_Avx512.cpp
    SourceFiles/MedianStack_Avx512.cpp
    SourceFiles/WarpPerspective_Avx512.cpp
)

if(MSVC)
    set(AVX2_FLAGS /arch:AVX2)
    set(AVX512_FLAGS /arch:AVX512)
else()
    set(AVX2_FLAGS -mavx2 -mfma)
    set(AVX512_FLAGS -mavx512f -mavx512bw -mavx512dq -mavx512vl -mfma)
endif()

set_source_files_properties(${TIERED_KERNEL_AVX2_SOURCES} PROPERTIES COMPILE_OPTIONS "${AVX2_FLAGS}")
set_source_files_properties(${TIERED_KERNEL_AVX512_SOURCES} PROPERTIES COMPILE_OPTIONS "${AVX512_FLAGS}")

# Vendored LibRaw's own file set (matches libraw/LibRaw.vcxproj's ClCompile
# list) - top-level src/*.cpp plus one level of subdirectories.
file(GLOB LIBRAW_SOURCES CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/libraw/src/*.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/libraw/src/*/*.cpp
)

# Vendored libjpeg-turbo (JpegCodec's CPU JPEG backend). Its own
# CMakeLists.txt hard-refuses add_subdirectory() ("use ExternalProject_Add()
# instead" - confirmed via a build spike), so it's built as a nested,
# independently-configured CMake project installed into the build tree,
# then linked as an IMPORTED STATIC target.
include(ExternalProject)

set(LIBJPEG_TURBO_INSTALL_DIR ${CMAKE_BINARY_DIR}/libjpeg-turbo-install)
set(LIBJPEG_TURBO_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}turbojpeg${CMAKE_STATIC_LIBRARY_SUFFIX}")

ExternalProject_Add(libjpeg_turbo_ext
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/libjpeg-turbo
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${LIBJPEG_TURBO_INSTALL_DIR}
        -DCMAKE_INSTALL_LIBDIR=lib
        -DCMAKE_BUILD_TYPE=Release
        -DENABLE_SHARED=FALSE
        -DENABLE_STATIC=TRUE
        -DWITH_TURBOJPEG=TRUE
    BUILD_BYPRODUCTS ${LIBJPEG_TURBO_INSTALL_DIR}/lib/${LIBJPEG_TURBO_LIB_NAME}
    INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
)

add_library(turbojpeg STATIC IMPORTED)
add_dependencies(turbojpeg libjpeg_turbo_ext)
set_target_properties(turbojpeg PROPERTIES
    IMPORTED_LOCATION "${LIBJPEG_TURBO_INSTALL_DIR}/lib/${LIBJPEG_TURBO_LIB_NAME}"
)
file(MAKE_DIRECTORY ${LIBJPEG_TURBO_INSTALL_DIR}/include)
target_include_directories(turbojpeg INTERFACE ${LIBJPEG_TURBO_INSTALL_DIR}/include)

add_library(windowsapp_core STATIC
    ${CORE_SOURCES}
    ${TIERED_KERNEL_SCALAR_SOURCES}
    ${TIERED_KERNEL_AVX2_SOURCES}
    ${TIERED_KERNEL_AVX512_SOURCES}
    ${LIBRAW_SOURCES}
    sqlite3/sqlite3.c
)

target_include_directories(windowsapp_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/HeaderFiles
        ${CMAKE_CURRENT_SOURCE_DIR}/sqlite3
        ${CMAKE_CURRENT_SOURCE_DIR}/libraw
        ${CMAKE_CURRENT_SOURCE_DIR}/libraw/internal
        ${CMAKE_CURRENT_SOURCE_DIR}/../WindowsApp.Compute
        ${CMAKE_CURRENT_SOURCE_DIR}/../WindowsApp.Compute/HeaderFiles
)

target_compile_definitions(windowsapp_core
    PRIVATE
        LIBRAW_NODLL
        SQLITE_OMIT_LOAD_EXTENSION
        SQLITE_THREADSAFE=1
)

target_link_libraries(windowsapp_core PRIVATE turbojpeg)

if(NOT WIN32)
    find_package(Threads REQUIRED)
    target_link_libraries(windowsapp_core PUBLIC Threads::Threads)
endif()
