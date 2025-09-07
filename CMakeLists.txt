cmake_minimum_required(VERSION 3.31)

project(audio_utils)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(./fetch_deps.cmake)

add_subdirectory(src)
add_subdirectory(tests)
