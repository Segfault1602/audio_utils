cmake_minimum_required(VERSION 3.31)

project(audio_utils)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(AUDIO_UTILS_BUILD_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})

include(./fetch_deps.cmake)

add_subdirectory(src)

if(AUDIO_UTILS_BUILD_TESTS)
    add_subdirectory(tests)
endif()
