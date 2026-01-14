#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <cerrno>
#include <iostream>
#include <atomic>

#include "TCPServer.hpp"
#include "InferenceEngine.hpp"
#include "IProcessor.hpp"
#include "InferenceTask.hpp"
#include "InferenceResult.hpp"
#include "ThreadSafeQueue.hpp"

extern std::atomic<bool> g_running;

namespace lumen {

ServerException::ServerException(const std::string& msg){
    full_msg = msg + ": " + std::strerror(errno);
}

const char* ServerException::what() const noexcept{
    return full_msg.c_str();
}

TCPServer::TCPServer(const char *port, ThreadSafeQueue<InferenceTask>& tq, ThreadSafeQueue<InferenceResult>& rq, InferenceEngine& e, std::shared_ptr<IPreProcessor> pr, std::shared_ptr<IPostProcessor> po) : task_queue(tq), response_queue(rq), engine(e), pre(pr), post(po){
    struct addrinfo hints = {}, *p, *res;
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    int status = getaddrinfo(nullptr, port, &hints, &res);
    if (status != 0) {
        gai_strerror(status);
    }

    for(p=res; p != nullptr; p = p->ai_next){
        if((sockfd = socket(p->ai_family, p->ai_socktype, p->ai_protocol)) == -1) continue;

        int opt = 1;
        setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        int flags = fcntl(sockfd, F_GETFL, 0);
        if(flags == -1 || fcntl(sockfd, F_SETFL, flags | O_NONBLOCK) == -1){
            close(sockfd);
            continue;
        }

        if(bind(sockfd, (sockaddr *)p->ai_addr, p->ai_addrlen) == -1){
            close(sockfd);
            continue;
        }

        break;

    }

    freeaddrinfo(res);
    if (p == nullptr) {
        throw ServerException("Failed to bind to any address");
    }

    if (listen(sockfd, backlog) == -1) {
        close(sockfd);
        throw ServerException("listen");
    }
    printf("[LumenEngine] Server started on port %s (Listening on fd: %d)\n", port, sockfd);

    struct pollfd listener_pfd;
    listener_pfd.fd = sockfd;
    listener_pfd.events = POLLIN;
    pollfds.push_back(listener_pfd);
}

TCPServer::~TCPServer(){
    printf("[LumenEngine] Shutting down server. Closing all connections...\n");
    for (auto& pfd : pollfds) {
        if (pfd.fd >= 0) {
            close(pfd.fd);
        }
    }
    printf("[LumenEngine] Offline.\n");
}

void TCPServer::handle_new_connection(){
    struct sockaddr_storage client_addr;
    socklen_t addr_size = sizeof(client_addr);

    int new_fd = accept(sockfd, (struct sockaddr *)&client_addr, &addr_size);
    if (new_fd == -1) {
        if (errno != EAGAIN && errno != EWOULDBLOCK) {
            perror("accept");
        }
    } else {
        int flags = fcntl(new_fd, F_GETFL, 0);
        if (flags == -1 || fcntl(new_fd, F_SETFL, flags | O_NONBLOCK) == -1) {
            perror("fcntl client");
            close(new_fd);
            return;
        }

        struct pollfd client_pfd;
        client_pfd.fd = new_fd;
        client_pfd.events = POLLIN;
        client_pfd.revents = 0;
        pollfds.push_back(client_pfd);

        printf("New connection accepted on fd %d\n", new_fd);
        sessions[new_fd] = {};
    }
}

int TCPServer::handle_client_data(int fd, size_t poll_idx){
    int bytes_received = 1;
    ClientSession& session = sessions[fd];
    int state = session.state;
    switch(state){
        case ClientSession::HEADER_PENDING:
            bytes_received = process_header(session, fd);
            break;
        case ClientSession::BODY_PENDING:
            bytes_received = process_body(session, fd);
            break;
        case ClientSession::READY_FOR_INFERENCE:
            bytes_received = finalize_request(session, fd);
            break;
        default:
            fprintf(stderr, "TCPServer: Unknown client session state found %d\n", state);
            break;
    }
    return bytes_received;
}

int TCPServer::process_header(ClientSession& session, int fd){
    int bytes_received = recv(fd, session.header_buffer + session.header_bytes_received, sizeof(session.header_buffer)-session.header_bytes_received, 0);

    if (bytes_received <= 0) {
        if (bytes_received == 0) {
            printf("Socket %d closed connection\n", fd);
        } else {
            perror("recv");
        }
    } else{
        session.header_bytes_received += bytes_received;
        if(session.header_bytes_received == sizeof(session.header_buffer)){
            session.state = ClientSession::BODY_PENDING;

            uint32_t network_size;
            std::memcpy(&network_size, session.header_buffer, 4);
            session.expected_size = ntohl(network_size);
            session.body_buffer.resize(session.expected_size);
            printf("Header processed. Expecting %u bytes at %p\n", session.expected_size, (void*)session.body_buffer.data());

            return 1;
        }
    }
    return bytes_received;
}

int TCPServer::process_body(ClientSession& session, int fd){
    if (session.body_buffer.size() <= 0) {
        fprintf(stderr, "Critical: Attempted to receive body into empty buffer!\n");
        return -1; 
    }
    int bytes_received = recv(fd, session.body_buffer.data() + session.bytes_received, session.expected_size-session.bytes_received, 0);

    if (bytes_received < 0) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return 1;
        }
        perror("recv body");
        return -1;
    } else if (bytes_received == 0) {
        printf("Socket %d closed connection during body\n", fd);
        return 0;
    }
    session.bytes_received += bytes_received;
    if(session.bytes_received == session.expected_size){
        session.state = ClientSession::READY_FOR_INFERENCE;
        printf("Body processed. Received %u bytes at %p\n", session.expected_size, (void*)session.body_buffer.data());
        finalize_request(session, fd);

        return 1;
    }
    return bytes_received;
}

int TCPServer::finalize_request(ClientSession& session, int fd){
    printf("Succesfully Received Tensor of %u bytes at %p\n", session.expected_size, (void*)session.body_buffer.data());
    
    /*
    FILE* fp = fopen("../tests/images/dump_from_network.bin", "wb");
    fwrite(session.data_ptr, 1, session.expected_size, fp);
    fclose(fp);
    */
    auto trace = std::make_unique<lumen::InferenceTrace>(fd);
    InferenceTask task(fd,
                       std::move(session.body_buffer),
                       pre, post,
                       std::move(trace));

    task_queue.push(std::move(task));
    session.state = ClientSession::INFERENCE_PENDING;
    return 1;
}

void TCPServer::close_connection(int fd, size_t poll_idx){
    close(pollfds[poll_idx].fd);
    pollfds[poll_idx] = pollfds.back();
    pollfds.pop_back();

    sessions.erase(fd);
}
void TCPServer::run() {
    while (g_running) {
        InferenceResult res;
        while (response_queue.try_pop_immediate(res)) {
            if (sessions.count(res.client_fd)) {
                ClientSession& session = sessions[res.client_fd];
                session.response_buffer = std::move(res.response);
                session.state = ClientSession::RESPONSE_SENDING;
                session.bytes_sent = 0;
            }
        }
        for(size_t i = 1; i < pollfds.size(); i++){
            switch(sessions[pollfds[i].fd].state){
                case ClientSession::RESPONSE_SENDING:
                    pollfds[i].events = POLLOUT;
                    break;
                case ClientSession::INFERENCE_PENDING:
                    pollfds[i].events = 0;
                    break;
                default:
                    pollfds[i].events = POLLIN;
                    break;
            }
        }
        int count = poll(pollfds.data(), pollfds.size(), 10);
        if (count == -1) {
            if (errno == EINTR) continue;
            throw ServerException("poll");
        }

        size_t current_size = pollfds.size();
        for (size_t i = 0; i < current_size; i++) {
            if (pollfds[i].revents & POLLIN) {
                int curr_fd = pollfds[i].fd;
                if (curr_fd == sockfd) {
                    handle_new_connection();
                }else{
                    int bytes_received;
                    if((bytes_received = handle_client_data(curr_fd, i)) <= 0){
                        close_connection(curr_fd, i);
                        --i;
                        --current_size;
                    }
                }
            } else if(pollfds[i].revents & POLLOUT){
                int curr_fd = pollfds[i].fd;
                int bytes_sent = 0;
                ClientSession& session = sessions[curr_fd];
                bytes_sent = send(curr_fd, session.response_buffer.c_str() + session.bytes_sent, session.response_buffer.size() - session.bytes_sent, 0);

                if(bytes_sent > 0){
                    session.bytes_sent += bytes_sent;
                    if(session.bytes_sent == session.response_buffer.size()){
                        session.state = ClientSession::HEADER_PENDING;
                        session.bytes_sent = 0;
                        session.response_buffer.clear();

                        session.expected_size = 0;
                        session.bytes_received = 0;
                        session.header_bytes_received = 0;
                        session.body_buffer.clear();
                    }
                }else if (bytes_sent == -1 && errno != EAGAIN && errno != EWOULDBLOCK) {
                    close_connection(curr_fd, i);
                    --i;
                    --current_size;
                }
            }
        }
    }
}
}
