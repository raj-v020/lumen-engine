import socket
import struct
import threading
import time
import random
import cv2
import numpy as np
import os

SERVER_IP = "127.0.0.1"
SERVER_PORT = 8080
# List of available test images
IMAGE_FILES = ["cat.jpg", "dog.jpg", "car.jpg", "bike.jpg"]
NUM_CLIENTS = 10

def send_request(client_id):
    try:
        # 1. Randomly select an image for this specific thread
        selected_image = random.choice(IMAGE_FILES)
        # Assuming images are in the engine/tests/images directory based on project structure
        image_path = selected_image

        # 2. Decode JPG to Raw Pixels
        img = cv2.imread(image_path)
        if img is None:
            print(f"[Client {client_id}] Error: Could not load {image_path}")
            return

        img = cv2.resize(img, (224, 224))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        payload = img_rgb.astype(np.uint8).tobytes() # 150,528 bytes
        
        # Simulate network jitter
        time.sleep(random.uniform(0, 0.5))
        
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((SERVER_IP, SERVER_PORT))
        
        # 3. Send Protocol Header + Raw Payload
        header = struct.pack("!I", len(payload))
        sock.sendall(header + payload)
        
        # 4. Receive and print result
        response = sock.recv(4096).decode().strip()
        print(f"[Client {client_id}] Sent {selected_image} | Received: {response}")
        
        sock.close()
    except Exception as e:
        print(f"[Client {client_id}] Error with {selected_image if 'selected_image' in locals() else 'unknown'}: {e}")

# Launch concurrent threads
threads = [threading.Thread(target=send_request, args=(i,)) for i in range(NUM_CLIENTS)]
for t in threads: t.start()
for t in threads: t.join()
