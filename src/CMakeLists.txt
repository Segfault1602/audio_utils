add_library(audio_utils)
add_library(audio_utils::audio_utils ALIAS audio_utils)

target_sources(
    audio_utils
    PRIVATE audio_analysis.cpp fft_utils.cpp array_math.cpp acoustics.cpp
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
           ./includes/audio_utils/ring_buffer.h
           ./includes/audio_utils/array_math.h)

target_link_libraries(audio_utils PRIVATE Eigen3::Eigen audioutils::audioutils_options
                                          audioutils::audioutils_warnings)

if(OpenMP_CXX_FOUND)
    target_link_libraries(audio_utils PRIVATE OpenMP::OpenMP_CXX)
endif()

if(IPP_FOUND)
    message(STATUS "Using Intel IPP for FFT computations")
    target_sources(audio_utils PRIVATE fft_ipp.cpp)
    target_compile_definitions(audio_utils PRIVATE AUDIO_UTILS_USE_IPP)
    target_link_libraries(audio_utils PRIVATE IPP::ippcore IPP::ipps IPP::ippvm)
else()
    target_sources(audio_utils PRIVATE fft.cpp)
    target_link_libraries(audio_utils PRIVATE pffft::pffft)
endif()

if(AUDIO_UTILS_USE_RTAUDIO)
    target_sources(audio_utils PRIVATE audio_manager.cpp rtaudio_impl.cpp test_tone.cpp)
    target_link_libraries(audio_utils PUBLIC RtAudio::rtaudio)
endif()

if(AUDIO_UTILS_USE_SNDFILE)
    target_sources(audio_utils PRIVATE audio_file_manager.cpp sndfile_manager_impl.cpp)
    target_link_libraries(audio_utils PRIVATE sndfile samplerate)
endif()

if(APPLE)
    target_link_libraries(audio_utils PRIVATE "-framework Accelerate")
endif()

target_compile_features(audio_utils PUBLIC cxx_std_23)
