#pragma once

#include <memory>
#include <string>
#include <vector>

#include "audio_file_manager.h"

using audio_stream_info = struct _audio_stream_info
{
    unsigned int sample_rate;
    unsigned int buffer_size;
    unsigned int num_input_channels;
    unsigned int num_output_channels;
};

class audio_manager
{
  public:
    static std::unique_ptr<audio_manager> create_audio_manager();

    audio_manager() = default;
    virtual ~audio_manager() = default;

    virtual bool start_audio_stream() = 0;
    virtual void stop_audio_stream() = 0;
    virtual bool is_audio_stream_running() const = 0;
    virtual audio_stream_info get_audio_stream_info() const = 0;
    virtual void select_input_channels(uint8_t channels) = 0;

    virtual void set_output_device(std::string_view device_name) = 0;
    virtual void set_input_device(std::string_view device_name) = 0;
    virtual void set_audio_driver(std::string_view driver_name) = 0;

    virtual std::vector<std::string> get_output_devices_name() const = 0;
    virtual std::vector<std::string> get_input_devices_name() const = 0;

    virtual std::vector<std::string> get_supported_audio_drivers() const = 0;
    virtual std::string get_current_audio_driver() const = 0;

    virtual void play_test_tone(bool play) = 0;

    virtual audio_file_manager* get_audio_file_manager() = 0;
};
