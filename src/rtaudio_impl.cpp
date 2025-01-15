#include "rtaudio_impl.h"

#include "audio_file_manager.h"
#include "sndfile_manager_impl.h"

#include <RtAudio.h>
#include <_string.h>
#include <cassert>
#include <iostream>
#include <vector>

namespace
{
void RtAudioErrorCb(RtAudioErrorType type, const std::string& errorText)
{
    std::cerr << "RTAudio Error: " << errorText << std::endl;
}
} // namespace

rtaudio_manager_impl::rtaudio_manager_impl()
    : current_output_device_id_(-1)
    , current_input_device_id_(-1)
{
    rtaudio_ = std::make_unique<RtAudio>(RtAudio::Api::MACOSX_CORE, RtAudioErrorCb);

    std::vector<RtAudio::Api> apis;
    rtaudio_->getCompiledApi(apis);
    for (auto api : apis)
    {
        std::cout << "Compiled API: " << RtAudio::getApiDisplayName(api) << std::endl;
    }

    // Don't open input by default
    current_output_device_id_ = rtaudio_->getDefaultOutputDevice();
    current_input_device_id_ = -1;

    audio_file_manager_ = std::make_unique<sndfile_manager_impl>();
}

rtaudio_manager_impl::~rtaudio_manager_impl()
{
    if (rtaudio_->isStreamOpen())
    {
        rtaudio_->closeStream();
    }
}

bool rtaudio_manager_impl::start_audio_stream()
{
    auto out_device_info = rtaudio_->getDeviceInfo(current_output_device_id_);
    RtAudio::StreamParameters out_parameters;
    out_parameters.deviceId = out_device_info.ID;
    out_parameters.nChannels = out_device_info.outputChannels;
    out_parameters.firstChannel = 0;

    auto in_device_info = rtaudio_->getDeviceInfo(current_input_device_id_);
    RtAudio::StreamParameters in_parameters;
    in_parameters.deviceId = in_device_info.ID;
    in_parameters.nChannels = 1; // Only mono for now
    in_parameters.firstChannel = input_selected_channels_;

    // TODO: make these configurable
    uint32_t buffer_frames = 512;

    RtAudioErrorType error =
        rtaudio_->openStream(&out_parameters, (in_parameters.deviceId != 0) ? &in_parameters : nullptr, RTAUDIO_FLOAT32,
                             sample_rate_, &buffer_frames, &rtaudio_cb_static, this);

    if (error != RTAUDIO_NO_ERROR)
    {
        std::cerr << "Failed to open audio stream: " << rtaudio_->getErrorText() << std::endl;
        return false;
    }

    error = rtaudio_->startStream();
    if (error != RTAUDIO_NO_ERROR)
    {
        std::cerr << "Failed to start audio stream: " << rtaudio_->getErrorText() << std::endl;
        rtaudio_->closeStream();
        return false;
    }

    output_stream_parameters_ = out_parameters;
    input_stream_parameters_ = in_parameters;
    buffer_size_ = buffer_frames;

    test_tone_.SetSampleRate(sample_rate_);

    std::cout << "Audio stream started" << std::endl;

    return true;
}

void rtaudio_manager_impl::stop_audio_stream()
{
    if (rtaudio_->isStreamRunning())
    {
        rtaudio_->stopStream();
    }

    if (rtaudio_->isStreamOpen())
    {
        rtaudio_->closeStream();
    }
}

bool rtaudio_manager_impl::is_audio_stream_running() const
{
    return rtaudio_->isStreamRunning();
}

audio_stream_info rtaudio_manager_impl::get_audio_stream_info() const
{
    audio_stream_info info = {0};
    if (current_input_device_id_ == -1)
    {
        return info;
    }

    auto input_device_info = rtaudio_->getDeviceInfo(current_input_device_id_);
    info.sample_rate = sample_rate_;
    info.buffer_size = buffer_size_;
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
            current_output_device_id_ = device;
            stop_audio_stream();
            start_audio_stream();
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
            current_input_device_id_ = device;
            stop_audio_stream();
            start_audio_stream();
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
            if (is_audio_stream_running())
            {
                stop_audio_stream();
            }

            rtaudio_ = std::make_unique<RtAudio>(api, RtAudioErrorCb);
            current_output_device_id_ = rtaudio_->getDefaultOutputDevice();
            current_input_device_id_ = rtaudio_->getDefaultInputDevice();
            current_audio_api_ = api;
            start_audio_stream();
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
    stop_audio_stream();
    start_audio_stream();
}

std::vector<std::string> rtaudio_manager_impl::get_output_devices_name() const
{
    std::vector<unsigned int> devices = rtaudio_->getDeviceIds();
    std::vector<std::string> device_names;

    // First device is 'none'
    device_names.emplace_back("None");

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

    // First device is 'none'
    device_names.emplace_back("None");

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

void rtaudio_manager_impl::play_test_tone(bool play)
{
    play_test_tone_ = play;
}

audio_file_manager* rtaudio_manager_impl::get_audio_file_manager()
{
    return audio_file_manager_.get();
}

int rtaudio_manager_impl::rtaudio_cb_static(void* output_buffer, void* input_buffer, unsigned int n_buffer_frames,
                                            double stream_time, RtAudioStreamStatus status, void* user_data)
{
    return static_cast<rtaudio_manager_impl*>(user_data)->rtaudio_cb_impl(output_buffer, input_buffer, n_buffer_frames,
                                                                          stream_time, status);
}

int rtaudio_manager_impl::rtaudio_cb_impl(void* output_buffer, void* input_buffer, unsigned int n_buffer_frames,
                                          double stream_time, RtAudioStreamStatus status)
{
    if (status & RTAUDIO_INPUT_OVERFLOW)
    {
        std::cerr << "Stream overflow detected!" << std::endl;
    }
    if (status & RTAUDIO_OUTPUT_UNDERFLOW)
    {
        std::cerr << "Stream underflow detected!" << std::endl;
    }

    float* output = static_cast<float*>(output_buffer);
    float test_tone = 0.f;
    float* input = static_cast<float*>(input_buffer);

    if (output)
    {
        memset(output, 0, n_buffer_frames * output_stream_parameters_.nChannels * sizeof(float));
        audio_file_manager_->process_block(output, n_buffer_frames, output_stream_parameters_.nChannels);
        // Just write silence for now
        for (auto i = 0; i < n_buffer_frames; i++)
        {
            float tick = 0.f;
            if (play_test_tone_)
            {
                test_tone = test_tone_.Tick();
            }

            for (auto j = 0; j < output_stream_parameters_.nChannels; j++)
            {
                output[(i * output_stream_parameters_.nChannels) + j] += test_tone + tick;
            }
        }
    }

    return 0;
}