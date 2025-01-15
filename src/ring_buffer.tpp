#pragma once
#include "ring_buffer.h"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <type_traits>

namespace
{
template <typename T>
void FreeBuffer(T* buffer)
{
    if (buffer != nullptr)
    {
        if (std::is_arithmetic<T>::value)
        {
#ifdef _WIN32
            _aligned_free(buffer);
#else
            free(buffer);
#endif
        }
        else
        {
            free(buffer);
        }
    }
}
} // namespace

template <typename T>
ring_buffer<T>::ring_buffer(size_t size)
{
    resize(size);
}

template <typename T>
ring_buffer<T>::~ring_buffer()
{
    FreeBuffer(buffer_);
}

template <typename T>
void ring_buffer<T>::resize(size_t size)
{
    max_size_ = size;

    FreeBuffer(buffer_);

    if (std::is_arithmetic<T>::value)
    {
        const auto byte_size = max_size_ * sizeof(T);
        const auto alignment = 64;
        const auto padded_size = ((byte_size + alignment - 1) / alignment) * alignment;
#ifdef _WIN32
        buffer_ = static_cast<T*>(_aligned_malloc(padded_size, alignment));
#else
        buffer_ = static_cast<T*>(aligned_alloc(alignment, padded_size));
#endif
        max_size_ = padded_size / sizeof(T);
    }
    else
    {
        buffer_ = static_cast<T*>(malloc(max_size_ * sizeof(T)));
    }

    read_index_ = 0;
    write_index_ = 0;
}

template <typename T>
size_t ring_buffer<T>::get_size() const
{
    return max_size_;
}

template <typename T>
size_t ring_buffer<T>::get_read_available() const
{
    if (overflow_flag_)
    {
        return max_size_;
    }
    return (max_size_ + write_index_ - read_index_) % max_size_;
}

template <typename T>
size_t ring_buffer<T>::get_write_available() const
{
    if (overflow_flag_)
    {
        return 0;
    }

    return max_size_ - get_read_available();
}

template <typename T>
void ring_buffer<T>::write(const T* data, size_t size)
{
    if (size == 0)
    {
        return;
    }

    size_t write_available = get_write_available();

    if (write_available == 0)
    {
        std::cerr << "ring_buffer::write: No space to write, dropping samples" << std::endl;
        return;
    }

    if (size > write_available)
    {
        std::cerr << "ring_buffer::write: No enough space to write, dropping samples" << std::endl;
        size = write_available;
    }

    // Check if we need to wrap around and write in two step
    size_t write_index = write_index_ % max_size_;
    if (write_index + size > max_size_)
    {
        size_t first_chunk_size = max_size_ - write_index;
        std::copy(data, data + first_chunk_size, buffer_ + write_index);
        std::copy(data + first_chunk_size, data + size, buffer_);
    }
    else
    {
        std::copy(data, data + size, buffer_ + write_index);
    }

    size_t new_write_index = (write_index_ + size) % max_size_;
    if (new_write_index == read_index_)
    {
        overflow_flag_ = true;
    }

    write_index_.store(new_write_index);
}

template <typename T>
void ring_buffer<T>::read(T* data, size_t& size)
{
    size_t read_available = get_read_available();

    if (read_available == 0)
    {
        // std::cerr << "ring_buffer::read: No data to read" << std::endl;
        size = 0;
        return;
    }

    if (size > read_available)
    {
        std::cerr << "ring_buffer::read: No enough data to read, reading only available data" << std::endl;
        size = read_available;
    }

    // Check if we need to wrap around and read in two step
    size_t read_index = read_index_ % max_size_;
    if (read_index + size > max_size_)
    {
        size_t first_chunk_size = max_size_ - read_index;
        std::copy(buffer_ + read_index, buffer_ + read_index + first_chunk_size, data);
        std::copy(buffer_, buffer_ + size - first_chunk_size, data + first_chunk_size);
    }
    else
    {
        std::copy(buffer_ + read_index, buffer_ + read_index + size, data);
    }

    read_index_.store((read_index_ + size) % max_size_);
    overflow_flag_ = false;
}

template <typename T>
void ring_buffer<T>::peek(T* data, size_t& size)
{
    size_t read_available = get_read_available();

    if (read_available == 0)
    {
        // std::cerr << "ring_buffer::read: No data to read" << std::endl;
        size = 0;
        return;
    }

    if (size > read_available)
    {
        std::cerr << "ring_buffer::read: No enough data to read, reading only available data" << std::endl;
        size = read_available;
    }

    // Check if we need to wrap around and read in two step
    size_t read_index = read_index_ % max_size_;
    if (read_index + size > max_size_)
    {
        size_t first_chunk_size = max_size_ - read_index;
        std::copy(buffer_ + read_index, buffer_ + read_index + first_chunk_size, data);
        std::copy(buffer_, buffer_ + size - first_chunk_size, data + first_chunk_size);
    }
    else
    {
        std::copy(buffer_ + read_index, buffer_ + read_index + size, data);
    }
}

template <typename T>
void ring_buffer<T>::reset()
{
    read_index_ = 0;
    write_index_ = 0;
}