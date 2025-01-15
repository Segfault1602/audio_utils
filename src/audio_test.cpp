#include <iostream>

#include "audio_manager.h"

int main()
{
    auto audio_manager = audio_manager::create_audio_manager();
    if (!audio_manager)
    {
        std::cerr << "Failed to create audio manager" << std::endl;
        return 1;
    }

    return 0;
}