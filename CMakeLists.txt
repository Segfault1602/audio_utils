cmake_minimum_required(VERSION 3.31)

project(audio_utils)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(AUDIO_UTILS_BUILD_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
option(AUDIO_UTILS_USE_SNDFILE "Use sndfile library for audio file I/O" ON)
option(AUDIO_UTILS_USE_RTAUDIO "Use RtAudio library for audio playback and recording" ON)

option(AUDIO_UTILS_ENABLE_HARDENING "Enable hardening" ON)
option(AUDIO_UTILS_USE_SANITIZER "Use AddressSanitizer" ON)

include(cmake/project-warnings.cmake)
include(cmake/project-options.cmake)

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
endif()

add_library(audioutils_warnings INTERFACE)
add_library(audioutils::audioutils_warnings ALIAS audioutils_warnings)
target_compile_options(audioutils_warnings INTERFACE ${AUDIO_UTILS_WARNINGS_CXX})

include(./fetch_deps.cmake)

add_subdirectory(src)

if(AUDIO_UTILS_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()
