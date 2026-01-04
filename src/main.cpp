#include <iostream>
#include "LinearAllocator.hpp"

using namespace std;

struct TensorMock{
    float data[100];
};
int main(){
    LinearAllocator la = LinearAllocator(1024*1024);
    for (int i = 0; i < 5; i++) {
        // 1. Simulate a request
        la.reset(); // Wipe memory for new request

        // 2. Allocate complex data
        int* request_id = la.alloc<int>();
        *request_id = i;

        TensorMock* t = la.alloc<TensorMock>();
        t->data[0] = 3.14f;

        // 3. Verify Memory Reuse
        // The address of request_id should be IDENTICAL every time.
        cout << "Req " << i << " | Address: " << request_id << " | Val: " << *request_id << endl;
    }
    return 0;

}
