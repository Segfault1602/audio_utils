#pragma once

#include <string>

class audio_file_manager
{
  public:
    enum class AudioPlayerState
    {
        kPlaying,
        kPaused,
        kStopRequested,
        kStopped
    };

    audio_file_manager() = default;
    virtual ~audio_file_manager() = default;

    virtual void set_sample_rate(int sample_rate) = 0;

    virtual bool open_audio_file(std::string_view file_name) = 0;
    virtual std::string get_open_file_name() const = 0;
    virtual bool is_file_open() const = 0;

    virtual void process_block(float* out_buffer, size_t frame_size, size_t num_channels, float gain = 1.f) = 0;

    virtual AudioPlayerState get_state() const = 0;
    virtual void play() = 0;
    virtual void pause() = 0;
    virtual void resume() = 0;
    virtual void stop() = 0;
};