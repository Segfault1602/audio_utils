set(AUDIOLIB_SOURCE
    audio_manager.cpp
    rtaudio_impl.cpp
    test_tone.cpp
    sndfile_manager_impl.cpp
    fft_utils.cpp
    )

add_library(audio_utils STATIC ${AUDIOLIB_SOURCE})
target_compile_options(audio_utils PRIVATE -fsanitize=address -fno-omit-frame-pointer)
target_link_libraries(audio_utils PRIVATE rtaudio sndfile pffft samplerate)
target_include_directories(audio_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../includes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_options(audio_utils PUBLIC -fsanitize=address)

add_executable(audio_test audio_test.cpp)
target_include_directories(audio_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../includes)
target_link_libraries(audio_test PRIVATE audio_utils)