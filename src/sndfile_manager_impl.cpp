#include "sndfile_manager_impl.h"

#include "samplerate.h"
#include "sndfile.h"

#include <cassert>
#include <iostream>
#include <span>
#include <thread>

namespace audio_utils::audio_file
{
void WriteWavFile(std::string_view filename, std::span<const float> buffer, int sample_rate)
{
    SF_INFO sf_info{};
    sf_info.channels = 1;
    sf_info.samplerate = sample_rate;
    sf_info.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    SNDFILE* file = sf_open(filename.data(), SFM_WRITE, &sf_info);
    if (!file)
    {
        std::cerr << "Failed to open file for writing: " << filename << std::endl;
        return;
    }

    sf_count_t written = sf_writef_float(file, buffer.data(), buffer.size() / sf_info.channels);
    if (written != static_cast<sf_count_t>(buffer.size() / sf_info.channels))
    {
        std::cerr << "Failed to write all samples to file: " << filename << std::endl;
    }

    sf_close(file);
}
} // namespace audio_utils::audio_file

void sndfile_manager_impl::set_sample_rate(int sample_rate)
{
    sample_rate_ = sample_rate;
}

bool sndfile_manager_impl::open_audio_file(std::string_view file_name)
{
    if (state_ == AudioPlayerState::kPlaying)
    {
        // That might be a bit harsh, but we don't support changing files while playing
        std::cerr << "Trying to close file while playing..." << std::endl;
    }

    state_ = AudioPlayerState::kStopped;
    if (file_)
    {
        sf_close(file_);
        file_ = nullptr;
    }

    file_ = sf_open(file_name.data(), SFM_READ, &file_info_);
    if (!file_)
    {
        return false;
    }

    if (file_info_.channels != 1)
    {
        std::cerr << "Only mono files are supported" << std::endl;
        return false;
    }

    file_name_ = file_name;

    if (file_info_.samplerate != sample_rate_)
    {
        std::cout << "Resampling file from " << file_info_.samplerate << " to " << sample_rate_ << std::endl;
        need_resample_ = true;
        if (src_state_)
        {
            src_delete(src_state_);
        }
        src_state_ = src_new(SRC_SINC_BEST_QUALITY, file_info_.channels, nullptr);
        if (!src_state_)
        {
            std::cerr << "Failed to create resampler" << std::endl;
            return false;
        }

        size_t input_size = file_info_.frames * file_info_.channels;

        std::vector<float> resample_buffer(file_info_.frames * file_info_.channels);
        size_t read = sf_readf_float(file_, resample_buffer.data(), file_info_.frames);
        if (read != file_info_.frames)
        {
            std::cerr << "Failed to read file for resampling" << std::endl;
            return false;
        }

        float ratio = static_cast<float>(sample_rate_) / file_info_.samplerate;

        size_t output_size = static_cast<size_t>(input_size * ratio) + 1;
        std::vector<float> out_buffer(output_size, 0);

        SRC_DATA src_data;
        src_data.data_in = resample_buffer.data();
        src_data.data_out = out_buffer.data();
        src_data.input_frames = file_info_.frames;
        src_data.output_frames = output_size;
        src_data.src_ratio = ratio;

        int error = src_simple(&src_data, SRC_SINC_BEST_QUALITY, file_info_.channels);

        if (error)
        {
            std::cerr << "Failed to resample file" << std::endl;
            return false;
        }

        if (src_data.input_frames_used != file_info_.frames)
        {
            std::cerr << "Not all input frames were used" << std::endl;
        }

        sf_close(file_);

        std::string resampled_file_name = std::string("resample_").append(file_name);
        file_ = sf_open(resampled_file_name.c_str(), SFM_WRITE, &file_info_);

        SF_INFO resampled_info{};
        resampled_info.channels = file_info_.channels;
        resampled_info.samplerate = sample_rate_;
        resampled_info.frames = src_data.output_frames_gen;
        resampled_info.format = file_info_.format;
        sf_writef_float(file_, out_buffer.data(), src_data.output_frames_gen);
        sf_write_sync(file_);
        sf_close(file_);

        file_ = sf_open(resampled_file_name.c_str(), SFM_READ, &file_info_);
        if (!file_)
        {
            std::cerr << "Failed to open resampled file" << std::endl;
            return false;
        }
    }

    std::cout << "Opened file: " << file_name << std::endl;
    std::cout << "Channels: " << file_info_.channels << std::endl;
    std::cout << "Sample rate: " << file_info_.samplerate << std::endl;
    std::cout << "Frames: " << file_info_.frames << std::endl;
    std::cout << "Format:" << file_info_.format << std::endl;

    return true;
}

std::string sndfile_manager_impl::get_open_file_name() const
{
    if (file_)
    {
        return file_name_;
    }

    return "";
}

bool sndfile_manager_impl::is_file_open() const
{
    return file_ != nullptr;
}

void sndfile_manager_impl::process_block(std::span<float> out_buffer, size_t frame_size, size_t num_channels,
                                         float gain)
{
    assert(out_buffer.size() >= frame_size * num_channels);
    if (state_ == AudioPlayerState::kStopped || state_ == AudioPlayerState::kPaused)
    {
        return;
    }

    if (state_ == AudioPlayerState::kStopRequested)
    {
        sf_seek(file_, 0, SEEK_SET);
        current_frame_ = 0;
        state_ = AudioPlayerState::kStopped;
        std::cout << "Audio playback stopped." << std::endl;
        return;
    }

    size_t total_size = frame_size * num_channels;
    if (buffer_.size() < total_size)
    {
        buffer_.resize(total_size);
    }

    size_t read = sf_readf_float(file_, buffer_.data(), frame_size);
    current_frame_ += read;

    if (read < frame_size && loop_)
    {
        assert(current_frame_ == file_info_.frames && "Current frame should match file frames when looping");
        sf_seek(file_, 0, SEEK_SET);
        size_t remaining = frame_size - read;
        size_t looped_read = sf_readf_float(file_, std::span(buffer_).subspan(read).data(), remaining);

        assert(looped_read == remaining && "Looped read did not match expected size");
        read += looped_read;
        current_frame_ = looped_read;
    }
    else if (read == 0)
    {
        state_ = AudioPlayerState::kStopRequested;
        return;
    }

    if (file_info_.channels == num_channels)
    {
        for (size_t i = 0; i < read; ++i)
        {
            for (size_t j = 0; j < num_channels; ++j)
            {
                out_buffer[(i * num_channels) + j] += buffer_[(i * num_channels) + j] * gain;
            }
        }
    }
    else
    {
        assert(file_info_.channels == 1);
        for (size_t i = 0; i < read; ++i)
        {
            for (size_t j = 0; j < num_channels; ++j)
            {
                out_buffer[(i * num_channels) + j] += buffer_[i];
            }
        }
    }
}

audio_file_manager::AudioPlayerState sndfile_manager_impl::get_state() const
{
    return state_;
}

void sndfile_manager_impl::play(bool loop)
{
    state_ = AudioPlayerState::kPlaying;
    if (file_ == nullptr && !file_name_.empty())
    {
        open_audio_file(file_name_);
    }
    loop_ = loop;
}

void sndfile_manager_impl::pause()
{
    state_ = AudioPlayerState::kPaused;
}

void sndfile_manager_impl::resume()
{
    state_ = AudioPlayerState::kPlaying;
}

void sndfile_manager_impl::stop(bool blocking)
{
    state_ = AudioPlayerState::kStopRequested;
    if (blocking)
    {
        while (state_ != AudioPlayerState::kStopped)
        {
            std::this_thread::yield();
        }
    }
}