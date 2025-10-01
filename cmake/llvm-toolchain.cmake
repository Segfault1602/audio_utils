include_guard(GLOBAL)

if(APPLE)
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm@20/bin/clang++")
    set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm@20/bin/clang")
else()
    set(CMAKE_CXX_COMPILER "clang++")
    set(CMAKE_C_COMPILER "clang")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(AUDIO_UTILS_SANITIZER -fsanitize=address)
    set(AUDIO_UTILS_COMPILE_DEFINITION -D_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_DEBUG)
endif()

set(AUDIO_UTILS_CXX_COMPILE_OPTIONS
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wno-sign-compare
    -Wunsafe-buffer-usage
    -fno-omit-frame-pointer
    ${AUDIO_UTILS_SANITIZER})

set(AUDIO_UTILS_LINK_OPTIONS ${AUDIO_UTILS_SANITIZER})

include($ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
