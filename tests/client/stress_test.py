import socket
import struct
import threading
import time
import random
import cv2
import numpy as np
import os

# --- CONFIGURATION ---
SERVER_IP = "127.0.0.1"
SERVER_PORT = 8080
IMAGE_DIR = "tests/images"  # Path to your images folder
IMAGE_NAMES = ["cat.jpg", "dog.jpg", "car.jpg", "bike.jpg"]
NUM_CLIENTS = 16            # Number of concurrent threads
REQUESTS_PER_CLIENT = 50    # How many requests each thread sends

# --- PRE-LOAD IMAGES ---
# We load and resize images ONCE into memory to avoid Disk I/O bottlenecks
# This ensures we are testing the Server's speed, not your Hard Drive's speed.
loaded_payloads = []

print("--- PREPARING DATA ---")
for img_name in IMAGE_NAMES:
    path = img_name
    if not os.path.exists(path):
        print(f"Warning: File not found {path}, skipping.")
        continue
        
    # 1. Decode JPG to Raw Pixels (BGR)
    img = cv2.imread(path)
    if img is None: 
        continue
        
    # 2. Resize to Model Input Size (224x224)
    img = cv2.resize(img, (224, 224))
    
    # 3. Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 4. Flatten to raw bytes (150,528 bytes)
    payload = img_rgb.astype(np.uint8).tobytes()
    loaded_payloads.append(payload)

if not loaded_payloads:
    print("Error: No images loaded. Check your tests/images/ folder.")
    exit(1)

print(f"Loaded {len(loaded_payloads)} images into memory. Each payload: {len(loaded_payloads[0])} bytes.")

# --- WORKER FUNCTION ---
def worker(client_id, results_list):
    success_count = 0
    
    for _ in range(REQUESTS_PER_CLIENT):
        try:
            # Pick a random raw payload from memory
            payload = random.choice(loaded_payloads)
            
            # Simulate slight client-side jitter (optional, removes "thundering herd")
            # time.sleep(random.uniform(0.001, 0.01))
            
            start_req = time.time()
            
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((SERVER_IP, SERVER_PORT))
            
            # 1. Send Protocol Header (4 bytes, Big Endian)
            header = struct.pack("!I", len(payload))
            sock.sendall(header + payload)
            
            # 2. Receive Result
            response = sock.recv(4096)
            sock.close()
            
            end_req = time.time()
            
            if response:
                success_count += 1
                results_list.append(end_req - start_req) # Record latency
                
        except Exception as e:
            print(f"[Client {client_id}] Error: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"\n--- STARTING STRESS TEST ---")
    print(f"Target: {SERVER_IP}:{SERVER_PORT}")
    print(f"Concurrency: {NUM_CLIENTS} threads")
    print(f"Total Requests: {NUM_CLIENTS * REQUESTS_PER_CLIENT}")
    
    latencies = [] # Shared list to store all latencies (Thread-safe-ish for append in Python)
    threads = []
    
    start_time = time.time()
    
    for i in range(NUM_CLIENTS):
        t = threading.Thread(target=worker, args=(i, latencies))
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
        
    end_time = time.time()
    total_time = end_time - start_time
    total_reqs = len(latencies)
    
    # Calculate Stats
    if total_reqs > 0:
        avg_lat = sum(latencies) / total_reqs * 1000
        latencies.sort()
        p99_lat = latencies[int(total_reqs * 0.99)] * 1000
        rps = total_reqs / total_time
        
        print(f"\n--- RESULTS ---")
        print(f"Total Time:      {total_time:.2f} s")
        print(f"Successful Reqs: {total_reqs}")
        print(f"Throughput:      {rps:.2f} RPS")
        print(f"Avg Latency:     {avg_lat:.2f} ms")
        print(f"P99 Latency:     {p99_lat:.2f} ms")
    else:
        print("Test failed: No requests completed.")
