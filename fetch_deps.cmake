
include(FetchContent)

FetchContent_Declare(
    rtaudio
    GIT_REPOSITORY https://github.com/thestk/rtaudio.git
    GIT_TAG 6.0.1
    )

set(RTAUDIO_API_ASIO OFF CACHE BOOL "" FORCE)
set(RTAUDIO_API_JACK OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(RTAUDIO_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(RTAUDIO_TARGETNAME_UNINSTALL "RTAUDIO_UNINSTALL" CACHE BOOL "RTAUDIO_UNINSTALL" FORCE)
FetchContent_MakeAvailable(rtaudio)

FetchContent_Declare(
    libsndfile
    GIT_REPOSITORY https://github.com/libsndfile/libsndfile.git
    GIT_TAG 1.2.2
    )

set(BUILD_PROGRAMS OFF CACHE BOOL "Don't build libsndfile programs!")
set(BUILD_EXAMPLES OFF CACHE BOOL "Don't build libsndfile examples!")
set(BUILD_REGTEST OFF CACHE BOOL "Don't build libsndfile regtest!")
set(BUILD_PROGRAMS OFF CACHE BOOL "Don't build libsndfile programs!" FORCE)
set(ENABLE_EXTERNAL_LIBS OFF CACHE BOOL "Disable external libs support!" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable libsndfile tests!" FORCE)
set(ENABLE_MPEG OFF CACHE BOOL "Disable MPEG support!" FORCE)
set(ENABLE_CPACK OFF CACHE BOOL "Disable CPACK!" FORCE)
set(ENABLE_PACKAGE_CONFIG OFF CACHE BOOL "Disable package config!" FORCE)
set(INSTALL_PKGCONFIG_MODULE OFF CACHE BOOL "Disable pkgconfig module!" FORCE)

FetchContent_MakeAvailable(libsndfile)

FetchContent_Declare(
    pffft
    GIT_REPOSITORY https://bitbucket.org/jpommier/pffft.git
)

FetchContent_MakeAvailable(pffft)
add_library(pffft STATIC ${pffft_SOURCE_DIR}/pffft.c)
target_compile_definitions(pffft PRIVATE -D_USE_MATH_DEFINES)
target_include_directories(pffft PUBLIC ${pffft_SOURCE_DIR})

FetchContent_Declare(
    libsamplerate
    GIT_REPOSITORY https://github.com/libsndfile/libsamplerate.git
)
FetchContent_MakeAvailable(libsamplerate)