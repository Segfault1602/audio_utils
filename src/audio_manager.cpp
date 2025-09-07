#include "audio_utils/audio_manager.h"

#include "rtaudio_impl.h"

std::unique_ptr<audio_manager> audio_manager::create_audio_manager()
{
    return std::make_unique<rtaudio_manager_impl>();
}