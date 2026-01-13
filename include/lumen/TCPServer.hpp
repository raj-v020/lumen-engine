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
#include "InferenceEngine.hpp"
#include "Processor.hpp"
#include "ThreadSafeQueue.hpp"
#include "InferenceTask.hpp"
#include "InferenceResult.hpp"

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
  InferenceEngine& engine;
  std::shared_ptr<IPreProcessor> pre;
  std::shared_ptr<IPostProcessor> post;
  ThreadSafeQueue<InferenceTask>& task_queue;
  ThreadSafeQueue<InferenceResult>& response_queue;

  struct ClientSession{
    enum State {
      HEADER_PENDING,
      BODY_PENDING,
      READY_FOR_INFERENCE,
      INFERENCE_PENDING,
      RESPONSE_SENDING
    };

    State state = HEADER_PENDING;

    uint32_t expected_size = 0;
    size_t bytes_received = 0;
    std::vector<uint8_t> body_buffer;

    unsigned char header_buffer[4];
    size_t header_bytes_received = 0;

    std::string response_buffer;
    size_t bytes_sent = 0;

    void reset() {
        body_buffer.clear();
        bytes_received = 0;
        expected_size = 0;
    }
  };


  std::unordered_map<int, ClientSession> sessions;
  void handle_new_connection();
  int handle_client_data(int fd, size_t poll_index);

  int process_header(ClientSession& session, int fd);
  int process_body(ClientSession& session, int fd);
  int finalize_request(ClientSession& session, int fd);

  void close_connection(int fd, size_t poll_index);

public:
  TCPServer(const char *port, ThreadSafeQueue<InferenceTask>& tq, ThreadSafeQueue<InferenceResult>& rq, Arena& a, InferenceEngine& engine, std::shared_ptr<IPreProcessor> pr, std::shared_ptr<IPostProcessor> po);
  ~TCPServer();
  void run();
};
}
