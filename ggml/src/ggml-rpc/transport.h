#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct socket_t;
typedef std::shared_ptr<socket_t> socket_ptr;

static constexpr size_t MAX_CHUNK_SIZE = 1024ull * 1024ull * 1024ull; // 1 GiB
static constexpr size_t RPC_CONN_CAPS_SIZE = 24;

struct socket_t {
    ~socket_t();

    bool send_data(const void * data, size_t size);
    bool recv_data(void * data, size_t size);

    // Append to this socket's send buffer instead of sending immediately; the buffer is flushed when it
    // reaches the GGML_RPC_BATCH_BYTES threshold (default 4096, 0 disables buffering), before any direct
    // send_data on this socket, and before any recv_data on ANY socket (see flush_all_pending).
    // Coalescing many small commands into one message avoids a per-message post+poll on RDMA transports
    // and a per-message syscall on TCP.
    bool send_data_buffered(const void * data, size_t size);
    bool flush_send();

    // Flush the send buffers of all sockets with pending data. Must run before any blocking receive:
    // a command buffered on socket A can be a prerequisite for the response awaited on socket B
    // (e.g. a pairwise allreduce between two RPC servers).
    static bool flush_all_pending();

    socket_ptr accept();

    void get_caps(uint8_t * local_caps);
    void update_caps(const uint8_t * remote_caps);

    static socket_ptr create_server(const char * host, int port);
    static socket_ptr connect(const char * host, int port);

    struct impl;

private:
    explicit socket_t(std::unique_ptr<impl> p);
    std::unique_ptr<impl> pimpl;
};

bool rpc_transport_init();
void rpc_transport_shutdown();
