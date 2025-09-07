add_library(audio_utils)
add_library(audio_utils::audio_utils ALIAS audio_utils)

target_sources(
    audio_utils
    PRIVATE audio_analysis.cpp
            audio_file_manager.cpp
            audio_manager.cpp
            fft_utils.cpp
            ring_buffer.h
            rtaudio_impl.cpp
            sndfile_manager_impl.cpp
            test_tone.cpp
    PUBLIC FILE_SET
           "audio_utils_headers"
           TYPE
           HEADERS
           BASE_DIRS
           ./includes
           FILES
           ./includes/audio_utils/audio_analysis.h
           ./includes/audio_utils/audio_file_manager.h
           ./includes/audio_utils/audio_manager.h
           ./includes/audio_utils/fft_utils.h)

target_link_libraries(
    audio_utils
    PRIVATE RtAudio::rtaudio
            SndFile::sndfile
            FFTW3::fftw3f
            SampleRate::samplerate
            Eigen3::Eigen)

target_compile_options(audio_utils PRIVATE $<$<CONFIG:Debug>:-fsanitize=address>
                                           -fno-omit-frame-pointer)

target_link_options(audio_utils PUBLIC $<$<CONFIG:Debug>:-fsanitize=address>)

target_compile_features(audio_utils PUBLIC cxx_std_20)

install(
    TARGETS audio_utils
    EXPORT audio_utilsTargets
    FILE_SET audio_utils_headers)

install(
    EXPORT audio_utilsTargets
    DESTINATION share/cmake/audio_utils
    NAMESPACE audio_utils::)
