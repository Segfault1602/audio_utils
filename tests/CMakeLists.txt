find_package(Catch2 REQUIRED)
find_package(nanobench REQUIRED)

add_executable(audio_test audio_test.cpp)
target_compile_options(audio_test PRIVATE ${AUDIO_UTILS_CXX_COMPILE_OPTIONS})
target_link_options(audio_test PUBLIC ${AUDIO_UTILS_LINK_OPTIONS})
target_compile_definitions(audio_test PRIVATE ${AUDIO_UTILS_COMPILE_DEFINITION})
target_link_libraries(audio_test PRIVATE audio_utils Catch2::Catch2WithMain SndFile::sndfile)

add_executable(audio_perf audio_perf.cpp)
target_link_libraries(audio_perf PRIVATE audio_utils Catch2::Catch2WithMain nanobench::nanobench
                                         SndFile::sndfile)
target_compile_options(audio_perf PRIVATE ${AUDIO_UTILS_CXX_COMPILE_OPTIONS})
target_link_options(audio_perf PUBLIC ${AUDIO_UTILS_LINK_OPTIONS})
target_compile_definitions(audio_perf PRIVATE ${AUDIO_UTILS_COMPILE_DEFINITION})
