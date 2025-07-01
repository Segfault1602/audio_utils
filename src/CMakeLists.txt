set(AUDIOLIB_SOURCE
    audio_manager.cpp
    audio_file_manager.cpp
    audio_metrics.cpp
    rtaudio_impl.cpp
    test_tone.cpp
    sndfile_manager_impl.cpp
    fft_utils.cpp
    )

add_library(audio_utils STATIC ${AUDIOLIB_SOURCE})
target_compile_options(audio_utils PRIVATE $<$<CONFIG:Debug>:-fsanitize=address> -fno-omit-frame-pointer)
target_link_libraries(audio_utils PRIVATE rtaudio sndfile fftw3f samplerate Eigen3::Eigen)
target_include_directories(audio_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../includes PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${FFTW3_INCLUDE_DIRS})
target_link_options(audio_utils PUBLIC $<$<CONFIG:Debug>:-fsanitize=address>)
target_link_directories(audio_utils PUBLIC ${FFTW3_LIBRARY_DIRS})

add_executable(audio_test audio_test.cpp)
target_include_directories(audio_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../includes)
target_link_libraries(audio_test PRIVATE audio_utils)