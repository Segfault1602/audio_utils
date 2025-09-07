# add_executable(
#     spectrogram_test
#     spectrogram_test.cpp)
# target_link_libraries(
#     spectrogram_test
#     PRIVATE audio_utils SndFile::sndfile)

add_executable(audio_test audio_test.cpp)

target_link_libraries(audio_test PRIVATE audio_utils)
