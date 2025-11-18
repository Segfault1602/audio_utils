#pragma once

#include <memory>
#include <span>
#include <string>

namespace audio_utils::audio_file
{
void WriteWavFile(std::string_view filename, std::span<const float> buffer, int sample_rate);
}

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

    static std::unique_ptr<audio_file_manager> create_audio_file_manager();

    audio_file_manager() = default;
    virtual ~audio_file_manager() = default;

    virtual void set_sample_rate(int sample_rate) = 0;

    virtual bool open_audio_file(std::string_view file_name) = 0;
    virtual std::string get_open_file_name() const = 0;
    virtual bool is_file_open() const = 0;

    virtual void process_block(std::span<float> out_buffer, size_t frame_size, size_t num_channels,
                               float gain = 1.f) = 0;

    virtual AudioPlayerState get_state() const = 0;
    virtual void play(bool loop) = 0;
    virtual void pause() = 0;
    virtual void resume() = 0;
    virtual void stop(bool blocking = false) = 0;
};