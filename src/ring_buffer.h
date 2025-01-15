#pragma once

#include <atomic>
#include <memory>

template <typename T>
class ring_buffer
{
  public:
    ring_buffer(size_t size = 32768);
    ~ring_buffer();

    void resize(size_t size);

    size_t get_size() const;
    size_t get_read_available() const;
    size_t get_write_available() const;

    void write(const T* data, size_t size);
    void read(T* data, size_t& size);
    void peek(T* data, size_t& size);

    void reset();

  private:
    size_t max_size_ = 0;
    std::atomic<size_t> read_index_ = 0;
    std::atomic<size_t> write_index_ = 0;
    std::atomic_bool overflow_flag_ = false;
    T* buffer_ = nullptr;
};

#include "ring_buffer.tpp"