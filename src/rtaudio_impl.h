#pragma once

#include <RtAudio.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <sndfile.h>
#include <string>
#include <string_view>

#include "audio_manager.h"
#include "audio_file_manager.h"
#include "test_tone.h"

class audio_file_manager;

class rtaudio_manager_impl : public audio_manager
{
  public:
    rtaudio_manager_impl();
    ~rtaudio_manager_impl() override;

    bool start_audio_stream() override;
    void stop_audio_stream() override;
    bool is_audio_stream_running() const override;
    audio_stream_info get_audio_stream_info() const override;

    void set_output_device(std::string_view device_name) override;
    void set_input_device(std::string_view device_name) override;
    void set_audio_driver(std::string_view driver_name) override;
    void select_input_channels(uint8_t channels) override;

    std::vector<std::string> get_output_devices_name() const override;
    std::vector<std::string> get_input_devices_name() const override;
    std::vector<std::string> get_supported_audio_drivers() const override;
    std::string get_current_audio_driver() const override;

    void play_test_tone(bool play) override;

    audio_file_manager* get_audio_file_manager() override;

  private:
    static int rtaudio_cb_static(void* output_buffer, void* input_buffer, unsigned int n_buffer_frames, double stream_time,
                                 RtAudioStreamStatus status, void* user_data);
    int rtaudio_cb_impl(void* output_buffer, void* input_buffer, unsigned int n_buffer_frames, double stream_time,
                        RtAudioStreamStatus status);

    std::unique_ptr<RtAudio> rtaudio_;
    RtAudio::StreamParameters output_stream_parameters_;
    RtAudio::StreamParameters input_stream_parameters_;

    int current_output_device_id_ = -1;
    int current_input_device_id_ = -1;

    uint8_t input_selected_channels_ = 0;

    uint32_t buffer_size_ = 512;
    uint32_t sample_rate_ = 48000;
    RtAudio::Api current_audio_api_ = RtAudio::Api::UNSPECIFIED;

    bool play_test_tone_ = false;

    TestToneGenerator test_tone_;

    std::unique_ptr<audio_file_manager> audio_file_manager_;
};