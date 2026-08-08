if(MSVC)
    set(MYPROJECT_WARNINGS_CXX /W3 /permissive-)
elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
    set(MYPROJECT_WARNINGS_CXX
        -Wall
        -Wextra
        -Wpedantic
        -Wno-sign-compare
        -Wno-language-extension-token
        -Wno-elaborated-enum-base
        -Wunsafe-buffer-usage)
endif()
