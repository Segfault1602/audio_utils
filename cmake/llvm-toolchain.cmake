include_guard(GLOBAL)

set(CMAKE_CXX_COMPILER "/Users/alex/llvm_install/bin/clang++")
set(CMAKE_C_COMPILER "/Users/alex/llvm_install/bin/clang")

set(SFFDN_SANITIZER $<$<CONFIG:Debug>:-fsanitize=address>)

set(SFFDN_CXX_COMPILE_OPTIONS
    -Wall
    -Wextra
    -Wpedantic
    -Werror
    -Wno-sign-compare
    # -Wunsafe-buffer-usage
    ${SFFDN_SANITIZER})

set(SFFDN_LINK_OPTIONS ${SFFDN_SANITIZER})

include($ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)
