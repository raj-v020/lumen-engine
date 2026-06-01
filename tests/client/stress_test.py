import os
import random
import socket
import struct
import sys
import threading
import time

import cv2
import numpy as np

# --- CONFIGURATION ---
SERVER_IP = "127.0.0.1"
SERVER_PORT = 8080
IMAGE_NAMES = ["cat.jpg", "dog.jpg", "car.jpg", "bike.jpg"]
NUM_CLIENTS = 16  # Number of concurrent threads
REQUESTS_PER_CLIENT = 50  # How many requests each thread sends

# --- PRE-LOAD IMAGES ---
loaded_payloads = []

print("--- PREPARING DATA ---")
for img_name in IMAGE_NAMES:
    path = img_name
    if not os.path.exists(path):
        print(f"Warning: File not found {path}, skipping.")
        continue

    img = cv2.imread(path)
    if img is None:
        continue

    img = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    payload = img_rgb.astype(np.uint8).tobytes()
    loaded_payloads.append(payload)

if not loaded_payloads:
    print(
        "Error: No images loaded. Ensure your image files are in the current working directory."
    )
    exit(1)

print(
    f"Loaded {len(loaded_payloads)} images into memory. Each payload: {len(loaded_payloads[0])} bytes."
)

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Parse matrix tags passed down from the orchestrator script
    raw_queue_tag = sys.argv[1] if len(sys.argv) > 1 else "MPMC"
    raw_alloc_tag = sys.argv[2] if len(sys.argv) > 2 else "LUMEN_ARENA"

    # --- STANDARDIZATION MAPS ---
    queue_map = {
        "NAIVE_MUTEX": "NAIVE_MUTEX",
        "naive_mutex": "NAIVE_MUTEX",
        "SPSC": "SPSC",
        "spsc": "SPSC",
        "MPMC": "MPMC",
        "mpmc": "MPMC",
        "BATCHED": "BATCHED",
        "batched": "BATCHED",
    }

    alloc_map = {
        "STANDARD": "Standard",
        "standard": "Standard",
        "LUMEN_ARENA": "Arena",
        "lumen_arena": "Arena",
    }

    queue_tag = queue_map.get(raw_queue_tag, raw_queue_tag.upper())
    alloc_tag = alloc_map.get(raw_alloc_tag, raw_alloc_tag)

    file_queue_str = raw_queue_tag.lower()
    file_alloc_str = (
        "lumen_arena"
        if raw_alloc_tag.upper() == "LUMEN_ARENA"
        else raw_alloc_tag.lower()
    )

    print("\n--- STARTING STRESS TEST ---")
    print(f"Target: {SERVER_IP}:{SERVER_PORT}")
    print(f"Concurrency: {NUM_CLIENTS} threads")
    print(f"Total Requests: {NUM_CLIENTS * REQUESTS_PER_CLIENT}")
    print(f"Matrix Metadata: [{queue_tag}] + [{alloc_tag}]")

    # Shared list to store tuples of:
    # (timestamp, total_round_trip_ms, connection_ms, bytes_sent)
    latencies_data = []

    def worker_wrapped(client_id):
        for _ in range(REQUESTS_PER_CLIENT):
            sock = None
            try:
                payload = random.choice(loaded_payloads)
                start_req = time.time()

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(3.0)

                # Measure raw TCP connection handshake time
                start_conn = time.time()
                sock.connect((SERVER_IP, SERVER_PORT))
                end_conn = time.time()
                conn_ms = (end_conn - start_conn) * 1000

                # Send Protocol Header (4 bytes, Big Endian) + Data Payload
                header = struct.pack("!I", len(payload))
                full_packet = header + payload
                bytes_sent = len(full_packet)

                sock.sendall(full_packet)

                # Receive Result
                response = sock.recv(4096)
                end_req = time.time()

                if response:
                    total_rt_ms = (end_req - start_req) * 1000
                    latencies_data.append(
                        (int(end_req), total_rt_ms, conn_ms, bytes_sent)
                    )
            except Exception:
                pass
            finally:
                if sock:
                    try:
                        sock.close()
                    except Exception:
                        pass

    threads = []
    start_time = time.time()

    for i in range(NUM_CLIENTS):
        t = threading.Thread(target=worker_wrapped, args=(i,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    end_time = time.time()
    total_time = end_time - start_time
    total_reqs = len(latencies_data)

    if total_reqs > 0:
        raw_latencies = [item[1] for item in latencies_data]
        raw_conn_times = [item[2] for item in latencies_data]

        avg_lat = sum(raw_latencies) / total_reqs
        avg_conn = sum(raw_conn_times) / total_reqs

        raw_latencies.sort()
        p99_lat = raw_latencies[int(total_reqs * 0.99)]
        rps = total_reqs / total_time

        print("\n--- RESULTS ---")
        print(f"Total Time:      {total_time:.2f} s")
        print(f"Successful Reqs: {total_reqs}")
        print(f"Throughput:      {rps:.2f} RPS")
        print(f"Avg Connection:  {avg_conn:.2f} ms")
        print(f"Avg RoundTrip:   {avg_lat:.2f} ms")
        print(f"P99 RoundTrip:   {p99_lat:.2f} ms")

        # --- EXPORT TO ROOT RESULTS DIRECTORY ---
        root_results_dir = "../../results/macro"
        os.makedirs(root_results_dir, exist_ok=True)
        filename = f"squeezenet_{file_queue_str}_{file_alloc_str}.csv"
        macro_csv_path = os.path.join(root_results_dir, filename)

        print(f"Exporting expanded macro metrics to {macro_csv_path}...")
        with open(macro_csv_path, "w") as f:
            # Expanded schema tracking connection layout alongside payload metrics
            f.write(
                "timestamp,request_id,queue_type,alloc_type,connection_ms,bytes_sent,total_round_trip_ms\n"
            )
            for idx, (ts, total_ms, conn_ms, b_sent) in enumerate(latencies_data):
                f.write(
                    f"{ts},{idx},{queue_tag},{alloc_tag},{conn_ms:.4f},{b_sent},{total_ms:.4f}\n"
                )
        print("Macro export complete.")
    else:
        print("Test failed: No requests completed successfully.")
