include(FetchContent)

FetchContent_Declare(
    CPM
    GIT_REPOSITORY https://github.com/cpm-cmake/CPM.cmake
    GIT_TAG v0.42.1)
FetchContent_MakeAvailable(CPM)
include(${cpm_SOURCE_DIR}/cmake/CPM.cmake)

if(AUDIO_UTILS_USE_RTAUDIO)
    cpmaddpackage(
        URI
        "gh:thestk/rtaudio#6.0.1"
        OPTIONS
        "RTAUDIO_API_ASIO OFF"
        "RTAUDIO_API_JACK OFF"
        "BUILD_SHARED_LIBS OFF"
        "RTAUDIO_BUILD_TESTING OFF"
        "RTAUDIO_TARGETNAME_UNINSTALL RTAUDIO_UNINSTALL")
    add_library(RtAudio::rtaudio ALIAS rtaudio)
endif()

if(AUDIO_UTILS_USE_SNDFILE)
    cpmaddpackage(
        NAME
        libsndfile
        GIT_TAG
        master
        GIT_REPOSITORY
        "https://github.com/libsndfile/libsndfile"
        OPTIONS
        "BUILD_PROGRAMS OFF"
        "BUILD_EXAMPLES OFF"
        "BUILD_REGTEST OFF"
        "ENABLE_EXTERNAL_LIBS OFF"
        "BUILD_TESTING OFF"
        "ENABLE_MPEG OFF"
        "ENABLE_CPACK OFF"
        "ENABLE_PACKAGE_CONFIG OFF"
        "INSTALL_PKGCONFIG_MODULE OFF")

    cpmaddpackage(
        NAME
        libsamplerate
        GIT_TAG
        master
        GIT_REPOSITORY
        "https://github.com/libsndfile/libsamplerate"
        OPTIONS
        "BUILD_PROGRAMS OFF"
        "BUILD_EXAMPLES OFF"
        "BUILD_REGTEST OFF"
        "ENABLE_EXTERNAL_LIBS OFF"
        "BUILD_TESTING OFF"
        "ENABLE_MPEG OFF"
        "ENABLE_CPACK OFF"
        "ENABLE_PACKAGE_CONFIG OFF"
        "INSTALL_PKGCONFIG_MODULE OFF")
endif()

# find_package(Eigen3 3.4 REQUIRED NO_MODULE)
cpmaddpackage(
    NAME
    Eigen
    GIT_TAG
    5.0.1
    GIT_REPOSITORY
    https://gitlab.com/libeigen/eigen)

# if(Eigen_ADDED)
#     add_library(Eigen INTERFACE IMPORTED)
#     target_include_directories(Eigen INTERFACE ${Eigen_SOURCE_DIR})
# endif()

cpmaddpackage(
    NAME
    pffft
    GIT_TAG
    master
    GIT_REPOSITORY
    https://bitbucket.org/jpommier/pffft/src/master/)

if(pffft_ADDED)
    add_library(pffft STATIC ${pffft_SOURCE_DIR}/pffft.c)
    add_library(pffft::pffft ALIAS pffft)
    target_compile_definitions(pffft PRIVATE -D_USE_MATH_DEFINES)
    target_include_directories(pffft PUBLIC ${pffft_SOURCE_DIR})
endif()

find_package(IPP)
find_package(OpenMP)
