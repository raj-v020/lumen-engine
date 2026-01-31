import socket
import struct
import time

def send_tensor(data_string):
    host = '127.0.0.1'
    port = 8080
    
    # Convert string to bytes
    payload = data_string.encode('utf-8')
    payload_size = len(payload)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            print(f"Connecting to {host}:{port}...")
            s.connect((host, port))

            # 1. Pack size into 4 bytes (! = Big Endian, I = Unsigned Int)
            header = struct.pack('!I', payload_size)
            
            # 2. Send Header
            s.sendall(header)
            print(f"Sent header: {payload_size} bytes")
            
            # Small delay to test your server's ability to handle split packets
            time.sleep(0.1) 

            # 3. Send Body
            s.sendall(payload)
            print(f"Sent payload: {data_string}")

            # 4. Receive Response
            response = s.recv(2)
            print(f"Server responded: {response.decode('utf-8')}")

    except Exception as e:
        print(f"Client Error: {e}")

if __name__ == "__main__":
    send_tensor("LumenEngineTest_Data")
