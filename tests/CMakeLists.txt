add_executable(spectrogram_test spectrogram_test.cpp)
target_link_libraries(spectrogram_test PRIVATE audio_utils sndfile)