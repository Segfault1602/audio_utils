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

string(TOUPPER "${AUDIO_UTILS_FFT_BACKEND}" AUDIO_UTILS_FFT_BACKEND_UPPER)
if(AUDIO_UTILS_FFT_BACKEND_UPPER STREQUAL "AUTO")
    if(IPP_FOUND)
        set(AUDIO_UTILS_SELECTED_FFT_BACKEND "IPP")
    elseif(APPLE)
        set(AUDIO_UTILS_SELECTED_FFT_BACKEND "VDSP")
    else()
        set(AUDIO_UTILS_SELECTED_FFT_BACKEND "PFFFT")
    endif()
else()
    set(AUDIO_UTILS_SELECTED_FFT_BACKEND "${AUDIO_UTILS_FFT_BACKEND_UPPER}")
endif()

if(AUDIO_UTILS_SELECTED_FFT_BACKEND STREQUAL "IPP")
    if(NOT IPP_FOUND)
        message(FATAL_ERROR "AUDIO_UTILS_FFT_BACKEND=IPP requested, but Intel IPP was not found")
    endif()
    message(STATUS "Using Intel IPP for FFT computations")
    target_sources(audio_utils PRIVATE fft_ipp.cpp)
    target_compile_definitions(audio_utils PRIVATE AUDIO_UTILS_USE_IPP)
    target_link_libraries(audio_utils PRIVATE IPP::ippcore IPP::ipps IPP::ippvm)
elseif(AUDIO_UTILS_SELECTED_FFT_BACKEND STREQUAL "VDSP")
    if(NOT APPLE)
        message(FATAL_ERROR "AUDIO_UTILS_FFT_BACKEND=VDSP is only supported on Apple platforms")
    endif()
    message(STATUS "Using Apple's vDSP for FFT computations")
    target_sources(audio_utils PRIVATE fft_vdsp.cpp)
    target_compile_definitions(audio_utils PRIVATE AUDIO_UTILS_USE_VDSP)
    target_link_libraries(audio_utils PRIVATE "-framework Accelerate")
elseif(AUDIO_UTILS_SELECTED_FFT_BACKEND STREQUAL "PFFFT")
    message(STATUS "Using PFFFT for FFT computations")
    target_sources(audio_utils PRIVATE fft.cpp)
    target_link_libraries(audio_utils PRIVATE pffft::pffft)
else()
    message(FATAL_ERROR "Unknown AUDIO_UTILS_FFT_BACKEND: ${AUDIO_UTILS_FFT_BACKEND}")
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
