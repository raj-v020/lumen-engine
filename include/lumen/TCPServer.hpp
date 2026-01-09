#pragma once
#include <exception>
#include <vector>
#include <string>
#include <poll.h>
#include <netdb.h>
#include <netinet/in.h>
#include <cstdint>
#include <unordered_map>
#include "Arena.hpp"

namespace lumen {
class ServerException : public std::exception{
private:
  std::string full_msg;
public:
  ServerException(const std::string& msg);
  virtual const char* what() const noexcept override;
};

class TCPServer{
private:
  int sockfd;
  std::vector<struct pollfd> pollfds;
  int backlog = 10;
  Arena& arena;
  struct ClientSession{
    enum State {
      HEADER_PENDING,
      BODY_PENDING,
      READY_FOR_INFERENCE,
      RESPONSE_SENDING
    };

    State state = HEADER_PENDING;

    uint32_t expected_size = 0;
    size_t bytes_received = 0;
    unsigned char* data_ptr = nullptr;

    unsigned char header_buffer[4];
    size_t header_bytes_received = 0;
  };
  std::unordered_map<int, ClientSession> sessions;
  void handle_new_connection();
  int handle_client_data(int fd, size_t poll_index);

  int process_header(ClientSession& session, int fd);
  int process_body(ClientSession& session, int fd);
  int finalize_request(ClientSession& session, int fd);

  void close_connection(int fd, size_t poll_index);

public:
  TCPServer(const char *port, Arena& a);
  ~TCPServer();
  void run();
};
}
