cpmaddpackage(
    NAME
    Catch2
    GITHUB_REPOSITORY
    catchorg/Catch2
    VERSION
    3.12.0)

cpmaddpackage(
    NAME
    nanobench
    GITHUB_REPOSITORY
    martinus/nanobench
    VERSION
    4.3.11)

get_target_property(_catch2_inc Catch2 INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(Catch2 SYSTEM INTERFACE ${_catch2_inc})

get_target_property(_nanobench_inc nanobench INTERFACE_INCLUDE_DIRECTORIES)
target_include_directories(nanobench SYSTEM INTERFACE ${_nanobench_inc})

add_executable(audio_test audio_test.cpp)
target_compile_options(audio_test PRIVATE ${AUDIO_UTILS_CXX_COMPILE_OPTIONS})
target_link_options(audio_test PUBLIC ${AUDIO_UTILS_LINK_OPTIONS})
target_compile_definitions(audio_test PRIVATE ${AUDIO_UTILS_COMPILE_DEFINITION})
target_link_libraries(
    audio_test
    PRIVATE audio_utils
            Catch2::Catch2WithMain
            SndFile::sndfile
            audioutils::audioutils_options
            audioutils::audioutils_warnings)
target_compile_features(audio_test PRIVATE cxx_std_23)

add_test(NAME audio_test COMMAND audio_test)

add_executable(audio_perf audio_perf.cpp)
target_link_libraries(
    audio_perf
    PRIVATE audio_utils
            Catch2::Catch2WithMain
            nanobench::nanobench
            SndFile::sndfile
            audioutils::audioutils_options
            audioutils::audioutils_warnings)
target_compile_options(audio_perf PRIVATE ${AUDIO_UTILS_CXX_COMPILE_OPTIONS})
target_link_options(audio_perf PUBLIC ${AUDIO_UTILS_LINK_OPTIONS})
target_compile_definitions(audio_perf PRIVATE ${AUDIO_UTILS_COMPILE_DEFINITION})
