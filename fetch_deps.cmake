if(AUDIO_UTILS_USE_RTAUDIO)
    find_package(RtAudio REQUIRED)
endif()

if(AUDIO_UTILS_USE_SNDFILE)
    find_package(SampleRate REQUIRED)
    find_package(SndFile REQUIRED)
endif()

find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(pffft REQUIRED)
find_package(IPP)
