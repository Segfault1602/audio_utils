#pragma once

#include "audio_utils/audio_file_manager.h"

#include <atomic>
#include <cstddef>
#include <samplerate.h>
#include <sndfile.h>
#include <string>
#include <string_view>
#include <vector>

class sndfile_manager_impl : public audio_file_manager
{
  public:
    sndfile_manager_impl() = default;
    ~sndfile_manager_impl() override = default;
    void set_sample_rate(int sample_rate) override;

    bool open_audio_file(std::string_view file_name) override;

    std::string get_open_file_name() const override;
    bool is_file_open() const override;

    void process_block(float* out_buffer, size_t frame_size, size_t num_channels, float gain = 1.f) override;

    AudioPlayerState get_state() const override;
    void play(bool loop) override;
    void pause() override;
    void resume() override;
    void stop(bool blocking) override;

  private:
    SNDFILE* file_ = nullptr;
    SF_INFO file_info_{};
    size_t current_frame_ = 0;
    std::atomic<bool> is_playing_ = false;
    bool is_paused_ = false;
    std::string file_name_;
    int sample_rate_ = 48000;
    bool need_resample_ = false;

    SRC_STATE* src_state_ = nullptr;

    std::vector<float> buffer_;

    AudioPlayerState state_ = AudioPlayerState::kStopped;

    bool loop_ = false;
};