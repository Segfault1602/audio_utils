#include "audio_utils/audio_file_manager.h"

#include "sndfile_manager_impl.h"

std::unique_ptr<audio_file_manager> audio_file_manager::create_audio_file_manager()
{
    return std::make_unique<sndfile_manager_impl>();
}