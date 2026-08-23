cmake_minimum_required(VERSION 3.31)

project(audio_utils)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(AUDIO_UTILS_BUILD_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
option(AUDIO_UTILS_USE_SNDFILE "Use sndfile library for audio file I/O" ${PROJECT_IS_TOP_LEVEL})
option(AUDIO_UTILS_USE_RTAUDIO "Use RtAudio library for audio playback and recording"
       ${PROJECT_IS_TOP_LEVEL})

option(AUDIO_UTILS_ENABLE_HARDENING "Enable hardening" ON)
option(AUDIO_UTILS_USE_SANITIZER "Use AddressSanitizer" OFF)
set(AUDIO_UTILS_FFT_BACKEND
    "AUTO"
    CACHE STRING "FFT backend: AUTO, IPP, PFFFT, or VDSP")
set_property(CACHE AUDIO_UTILS_FFT_BACKEND PROPERTY STRINGS AUTO IPP PFFFT VDSP)

include(./fetch_deps.cmake)
include(cmake/project-warnings.cmake)
include(cmake/project-options.cmake)

if(AUDIO_UTILS_USE_RTAUDIO)
    target_compile_definitions(
        rtaudio PRIVATE $<TARGET_PROPERTY:audioutils_options,INTERFACE_COMPILE_DEFINITIONS>)
endif()

# if(PROJECT_IS_TOP_LEVEL)
#     set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
# endif()

add_library(audioutils_warnings INTERFACE)
add_library(audioutils::audioutils_warnings ALIAS audioutils_warnings)
target_compile_options(audioutils_warnings INTERFACE ${AUDIO_UTILS_WARNINGS_CXX})

add_subdirectory(src)

if(AUDIO_UTILS_BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()
