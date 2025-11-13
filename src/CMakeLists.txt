add_library(audio_utils)
add_library(audio_utils::audio_utils ALIAS audio_utils)

target_sources(
    audio_utils
    PRIVATE audio_analysis.cpp
            audio_file_manager.cpp
            audio_manager.cpp
            fft.cpp
            fft_utils.cpp
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
           ./includes/audio_utils/fft_utils.h
           ./includes/audio_utils/fft.h
           ./includes/audio_utils/ring_buffer.h)

target_link_libraries(
    audio_utils
    PUBLIC RtAudio::rtaudio
    PRIVATE
            SndFile::sndfile
            SampleRate::samplerate
            Eigen3::Eigen
            pffft::pffft)

if(APPLE)
    target_link_libraries(audio_utils PRIVATE "-framework Accelerate")
endif()

target_compile_options(audio_utils PRIVATE ${AUDIO_UTILS_CXX_COMPILE_OPTIONS})
target_link_options(audio_utils PUBLIC ${AUDIO_UTILS_LINK_OPTIONS})
target_compile_definitions(audio_utils PRIVATE ${AUDIO_UTILS_COMPILE_DEFINITION})

target_compile_features(audio_utils PUBLIC cxx_std_23)

install(
    TARGETS audio_utils
    EXPORT audio_utilsTargets
    FILE_SET audio_utils_headers)

install(
    EXPORT audio_utilsTargets
    DESTINATION share/cmake/audio_utils
    NAMESPACE audio_utils::)
