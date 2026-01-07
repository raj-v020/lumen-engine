#include <iostream>
#include "Arena.hpp"
#include "TCPServer.hpp"

using namespace std;

int main(){
    try{
        Arena arena(1024*1024);
        TCPServer server("8080", arena);
        server.run();
        cout << "Lumen Engine starting on port 8080..." << endl;
    }catch(const ServerException& e){
        cerr << "Server Error: " << e.what() << endl;
    }catch(const exception& e){
        cerr << "General Error: " << e.what() << endl;
    }
    return 0;

}
