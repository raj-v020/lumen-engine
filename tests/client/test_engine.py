import socket
import struct
import sys
import cv2
import numpy as np

def send_image_to_lumen(image_path, host='127.0.0.1', port=8080):
    # 1. Load Image
    img = cv2.imread(image_path)
    if img is None:
        print(f"[-] Error: Could not load image at {image_path}")
        return

    # 2. Pre-process (Match the Arena expectations)
    # We resize to 224x224 and convert to RGB (OpenCV is BGR by default)
    img = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Serialize to Raw Bytes
    # uint8 uses 1 byte per color channel. 224 * 224 * 3 = 150,528 bytes.
    payload = img_rgb.astype(np.uint8).tobytes()
    payload_size = len(payload) 

    try:
        # 4. Setup Connection
        # timeout=10 prevents the client from waiting forever if the server locks up
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(10.0) 
            print(f"[*] Connecting to LumenEngine at {host}:{port}...")
            s.connect((host, port))

            # 5. Send Header (The 4-byte 'contract')
            # '!I' ensures Network Byte Order (Big-Endian)
            header = struct.pack('!I', payload_size)
            s.sendall(header)
            
            # 6. Send Body (The actual Tensor data)
            s.sendall(payload)
            print(f"[*] Data sent ({payload_size} bytes). Waiting for ThreadPool...")

            # 7. Receive Result
            # We receive up to 4096 bytes for the result string
            response = s.recv(4096)
            
            if response:
                result_text = response.decode('utf-8')
                print("\n" + "="*40)
                print(f" LUMEN INFERENCE RESULT ")
                print("-" * 40)
                print(result_text)
                print("="*40 + "\n")
            else:
                print("[-] Server closed connection abruptly.")

    except socket.timeout:
        print("[-] Error: Request timed out. Is the ThreadPool stuck?")
    except ConnectionRefusedError:
        print("[-] Error: Could not connect. Is the TCPServer running?")
    except Exception as e:
        print(f"[-] Unexpected Client Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 lumen_client.py <image_path>")
    else:
        send_image_to_lumen(sys.argv[1])
