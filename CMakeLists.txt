cmake_minimum_required(VERSION 3.31)
set(CMAKE_CXX_STANDARD 20)

project(audio_utils)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


if (APPLE)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/src")
endif()
include(./fetch_deps.cmake)

add_subdirectory(src)
add_subdirectory(tests)
