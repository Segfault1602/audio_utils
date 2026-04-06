# audio_utils

A modern C++ audio utility library focused on analysis, FFT workflows, and optional real-time playback/file I/O support.

## Highlights

- FFT wrapper with real-forward, inverse, magnitude/power spectrum, cepstrum, and convolution support
- Analysis utilities including:
    - Autocorrelation
    - STFT
    - Mel spectrogram
    - Spectral flatness
    - Energy Decay Curve (EDC)
    - Energy Decay Relief (EDR)
    - T60 estimation
    - Echo density metrics
- Windowing helpers (Rectangular, Hamming, Hann, Blackman)
- Optional audio file read/write and playback/streaming abstractions
- Optional optimized FFT backends:
    - Intel IPP when available
    - Apple vDSP on macOS
    - PFFFT fallback elsewhere
    - Add your own FFT backend by implementing the FFT interface
    - `audio_utils::array_math` provides efficient vectorized implementation of common array operations (e.g. element-wise multiplication, scaling)

## Audio Playback

- Cross-platform real-time audio support via RtAudio (optional dependency)
- Render only for the moment

## Audio File I/O
- Optional libsndfile-based audio file reading/writing
- Automatically resamples to the target sample rate using libsamplerate when reading files

## Requirements

- C++23 compiler
- CMake 3.31+
- Ninja (recommended for preset workflows)

Dependencies are fetched automatically with CPM/FetchContent.

## Build


```bash
cmake --preset llvm-ninja
cmake --build --preset llvm-debug
```

Other build presets:

- llvm-release
- llvm-relwithdebinfo

## CMake Options

- AUDIO_UTILS_BUILD_TESTS: Build test and benchmark targets
- AUDIO_UTILS_USE_SNDFILE: Enable libsndfile/libsamplerate-based audio file I/O
- AUDIO_UTILS_USE_RTAUDIO: Enable RtAudio-based real-time audio support
- AUDIO_UTILS_ENABLE_HARDENING: Enable hardening flags from project options
- AUDIO_UTILS_USE_SANITIZER: Enable AddressSanitizer configuration


## Using in Your Project

This repository defines:

- Library target: audio_utils
- Alias target: audio_utils::audio_utils

