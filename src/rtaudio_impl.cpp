#include "rtaudio_impl.h"

#include <RtAudio.h>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <ranges>
#include <string>
#include <utility>
#include <vector>

namespace
{
void RtAudioErrorCb(RtAudioErrorType type, const std::string& errorText)
{
    std::cerr << "RTAudio Error (Type=" << static_cast<int>(type) << "): " << errorText << std::endl;
}
} // namespace

rtaudio_manager_impl::rtaudio_manager_impl()
{
    rtaudio_ = std::make_unique<RtAudio>(RtAudio::Api::MACOSX_CORE, RtAudioErrorCb);

    // std::vector<RtAudio::Api> apis;
    // rtaudio_->getCompiledApi(apis);
    // for (auto api : apis)
    // {
    //     std::cout << "Compiled API: " << RtAudio::getApiDisplayName(api) << std::endl;
    // }

    // Don't open input by default
    current_output_device_id_ = rtaudio_->getDefaultOutputDevice();
    current_input_device_id_ = rtaudio_->getDefaultInputDevice();
}

rtaudio_manager_impl::~rtaudio_manager_impl()
{
    stop_audio_stream();
}

bool rtaudio_manager_impl::start_audio_stream(audio_stream_option option, audio_cb cb, uint32_t block_size)
{
    stop_audio_stream();

    auto out_device_info = rtaudio_->getDeviceInfo(current_output_device_id_);
    RtAudio::StreamParameters out_parameters{};
    out_parameters.deviceId = out_device_info.ID;
    out_parameters.nChannels = out_device_info.outputChannels;
    out_parameters.firstChannel = 0;

    auto in_device_info = rtaudio_->getDeviceInfo(current_input_device_id_);
    RtAudio::StreamParameters in_parameters{};
    in_parameters.deviceId = in_device_info.ID;
    in_parameters.nChannels = 1; // Only mono for now
    in_parameters.firstChannel = input_selected_channels_;

    block_size_ = block_size;
    current_stream_option_ = option;
    output_stream_parameters_ = out_parameters;
    input_stream_parameters_ = in_parameters;
    cb_ = std::move(cb);
    test_tone_.SetSampleRate(sample_rate_);
    callback_enabled_.store(true, std::memory_order_release);

    RtAudio::StreamParameters* out_param_ptr = (option & audio_stream_option::kOutput) ? &out_parameters : nullptr;
    RtAudio::StreamParameters* in_param_ptr = (option & audio_stream_option::kInput) ? &in_parameters : nullptr;

    RtAudioErrorType error = rtaudio_->openStream(out_param_ptr, in_param_ptr, RTAUDIO_FLOAT32, sample_rate_,
                                                  &block_size_, &rtaudio_cb_static, this);

    if (error != RTAUDIO_NO_ERROR)
    {
        std::cerr << "Failed to open audio stream: " << rtaudio_->getErrorText() << std::endl;
        callback_enabled_.store(false, std::memory_order_release);
        cb_ = nullptr;
        return false;
    }

    error = rtaudio_->startStream();
    if (error != RTAUDIO_NO_ERROR)
    {
        std::cerr << "Failed to start audio stream: " << rtaudio_->getErrorText() << std::endl;
        stop_audio_stream();
        return false;
    }

    if (block_size_ != block_size)
    {
        std::cerr << "Warning: Block size mismatch. Expected: " << block_size << ", Actual: " << block_size_
                  << std::endl;
    }

    return true;
}

void rtaudio_manager_impl::stop_audio_stream()
{
    callback_enabled_.store(false, std::memory_order_release);

    if (rtaudio_->isStreamRunning())
    {
        rtaudio_->stopStream();
    }

    if (rtaudio_->isStreamOpen())
    {
        rtaudio_->closeStream();
    }

    uint32_t callbacks_in_flight = callbacks_in_flight_.load(std::memory_order_acquire);
    while (callbacks_in_flight != 0)
    {
        callbacks_in_flight_.wait(callbacks_in_flight, std::memory_order_acquire);
        callbacks_in_flight = callbacks_in_flight_.load(std::memory_order_acquire);
    }
    cb_ = nullptr;
}

bool rtaudio_manager_impl::is_audio_stream_running() const
{
    return rtaudio_->isStreamRunning();
}

audio_stream_info rtaudio_manager_impl::get_audio_stream_info() const
{
    audio_stream_info info{0, 0, 0, 0};
    if (current_input_device_id_ == -1)
    {
        return info;
    }

    auto input_device_info = rtaudio_->getDeviceInfo(current_input_device_id_);
    info.sample_rate = sample_rate_;
    info.buffer_size = block_size_;
    info.num_input_channels = input_device_info.inputChannels;
    info.num_output_channels = output_stream_parameters_.nChannels;
    return info;
}

void rtaudio_manager_impl::set_output_device(std::string_view device_name)
{
    if (device_name == "None")
    {
        stop_audio_stream();
        return;
    }

    auto devices = rtaudio_->getDeviceIds();
    for (auto device : devices)
    {
        auto info = rtaudio_->getDeviceInfo(device);
        if (info.name == device_name && device != current_output_device_id_ && info.outputChannels > 0)
        {
            assert(info.outputChannels > 0);
            const auto stream_option = current_stream_option_;
            const auto callback = cb_;
            const auto block_size = block_size_;
            current_output_device_id_ = device;

            if (rtaudio_->isStreamRunning())
            {
                stop_audio_stream();
                start_audio_stream(stream_option, callback, block_size);
            }
            return;
        }
    }
}

void rtaudio_manager_impl::set_input_device(std::string_view device_name)
{
    if (device_name == "None")
    {
        stop_audio_stream();
        return;
    }

    auto devices = rtaudio_->getDeviceIds();
    for (auto device : devices)
    {
        auto info = rtaudio_->getDeviceInfo(device);
        if (info.name == device_name && device != current_input_device_id_ && info.inputChannels > 0)
        {
            assert(info.inputChannels > 0);
            const auto stream_option = current_stream_option_;
            const auto callback = cb_;
            const auto block_size = block_size_;
            current_input_device_id_ = device;
            if (rtaudio_->isStreamRunning())
            {
                stop_audio_stream();
                start_audio_stream(stream_option, callback, block_size);
            }
            return;
        }
    }
}

void rtaudio_manager_impl::set_audio_driver(std::string_view driver_name)
{
    std::vector<RtAudio::Api> apis;
    RtAudio::getCompiledApi(apis);
    for (auto api : apis)
    {
        if (RtAudio::getApiDisplayName(api) == driver_name && api != current_audio_api_)
        {
            bool should_restart = false;
            const auto stream_option = current_stream_option_;
            const auto callback = cb_;
            const auto block_size = block_size_;
            if (is_audio_stream_running())
            {
                stop_audio_stream();
                should_restart = true;
            }

            rtaudio_ = std::make_unique<RtAudio>(api, RtAudioErrorCb);
            current_output_device_id_ = rtaudio_->getDefaultOutputDevice();
            current_input_device_id_ = rtaudio_->getDefaultInputDevice();
            current_audio_api_ = api;

            if (should_restart)
            {
                start_audio_stream(stream_option, callback, block_size);
            }
            return;
        }
    }
}

void rtaudio_manager_impl::select_input_channels(uint8_t channels)
{
    if (input_selected_channels_ == channels)
    {
        return;
    }

    input_selected_channels_ = channels;
    if (rtaudio_->isStreamRunning())
    {
        const auto stream_option = current_stream_option_;
        const auto callback = cb_;
        const auto block_size = block_size_;
        stop_audio_stream();
        start_audio_stream(stream_option, callback, block_size);
    }
}

std::vector<std::string> rtaudio_manager_impl::get_output_devices_name() const
{
    std::vector<unsigned int> devices = rtaudio_->getDeviceIds();
    std::vector<std::string> device_names;

    for (unsigned int i = 0; i < devices.size(); ++i)
    {
        auto info = rtaudio_->getDeviceInfo(devices[i]);
        if (info.outputChannels > 0)
        {
            device_names.push_back(info.name);
        }
    }

    return device_names;
}

std::vector<std::string> rtaudio_manager_impl::get_input_devices_name() const
{
    std::vector<unsigned int> devices = rtaudio_->getDeviceIds();
    std::vector<std::string> device_names;

    for (unsigned int i = 0; i < devices.size(); ++i)
    {
        auto info = rtaudio_->getDeviceInfo(devices[i]);
        if (info.inputChannels > 0)
        {
            device_names.push_back(info.name);
        }
    }

    return device_names;
}

std::vector<std::string> rtaudio_manager_impl::get_supported_audio_drivers() const
{
    std::vector<std::string> drivers;
    std::vector<RtAudio::Api> apis;
    RtAudio::getCompiledApi(apis);
    for (auto api : apis)
    {
        drivers.push_back(RtAudio::getApiDisplayName(api));
    }
    return drivers;
}

std::string rtaudio_manager_impl::get_current_audio_driver() const
{
    return RtAudio::getApiDisplayName(rtaudio_->getCurrentApi());
}

std::string rtaudio_manager_impl::get_current_output_device_name() const
{
    RtAudio::DeviceInfo output_device_info = rtaudio_->getDeviceInfo(current_output_device_id_);
    if (output_device_info.ID == 0)
    {
        return "None";
    }
    return output_device_info.name;
}

void rtaudio_manager_impl::play_test_tone(bool play)
{
    play_test_tone_.store(play, std::memory_order_relaxed);
}

int rtaudio_manager_impl::rtaudio_cb_static(void* output_buffer, void* input_buffer, unsigned int n_buffer_frames,
                                            double stream_time, RtAudioStreamStatus status, void* user_data)
{
    return static_cast<rtaudio_manager_impl*>(user_data)->rtaudio_cb_impl(output_buffer, input_buffer, n_buffer_frames,
                                                                          stream_time, status);
}

int rtaudio_manager_impl::rtaudio_cb_impl(void* output_buffer, [[maybe_unused]] void* input_buffer,
                                          unsigned int n_buffer_frames, [[maybe_unused]] double stream_time,
                                          RtAudioStreamStatus status)
{
    callbacks_in_flight_.fetch_add(1, std::memory_order_acq_rel);
    struct CallbackScope
    {
        std::atomic<uint32_t>& callbacks_in_flight;

        ~CallbackScope()
        {
            if (callbacks_in_flight.fetch_sub(1, std::memory_order_acq_rel) == 1)
            {
                callbacks_in_flight.notify_all();
            }
        }
    } callback_scope{callbacks_in_flight_};

    const uint32_t output_size = n_buffer_frames * output_stream_parameters_.nChannels;

#pragma clang unsafe_buffer_usage begin
    auto output_span = std::span(static_cast<float*>(output_buffer), output_size);
#pragma clang unsafe_buffer_usage end

    float* output = static_cast<float*>(output_buffer);
    float test_tone = 0.f;

    if (output)
    {
        std::ranges::fill(output_span, 0.0f);
    }

    if (!callback_enabled_.load(std::memory_order_acquire))
    {
        return 0;
    }

    if (status & RTAUDIO_INPUT_OVERFLOW)
    {
        std::cerr << "Stream overflow detected!" << std::endl;
    }
    if (status & RTAUDIO_OUTPUT_UNDERFLOW)
    {
        std::cerr << "Stream underflow detected!" << std::endl;
    }

    if (output)
    {
        if (cb_)
        {
            cb_(output_span, n_buffer_frames, output_stream_parameters_.nChannels);
        }

        if (play_test_tone_.load(std::memory_order_relaxed))
        {
            for (auto i = 0; i < n_buffer_frames; i++)
            {
                test_tone = test_tone_.Tick();
                for (auto j = 0; j < output_stream_parameters_.nChannels; j++)
                {
                    output_span[(i * output_stream_parameters_.nChannels) + j] += test_tone;
                }
            }
        }
    }

    return 0;
}