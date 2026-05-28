import json
import os
import shutil
import signal
import subprocess
import time

# --- FULL MATRIX CONFIGURATION ---
QUEUE_TYPES = ["NAIVE_MUTEX", "SPSC", "MPMC", "BATCHED"]
ALLOCATOR_TYPES = ["STANDARD", "LUMEN_ARENA"]

SERVER_BINARY = "./lumen_engine"
CLIENT_SCRIPT = "./stress_test.py"
CONFIG_PATH = "config.json"


def cleanup_and_backup_environment():
    """Purges lingering processes, backs up old metrics outside the root, and flushes the results dir."""
    print("[Orchestrator] Running system process sanitation step...")
    try:
        subprocess.run(
            ["killall", "-9", "lumen_engine"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
    try:
        subprocess.run(
            ["pkill", "-9", "-f", "stress_test.py"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
    time.sleep(1.0)

    backup_dir = "../lumen_telemetry_backup"
    source_results = "results"

    if os.path.exists(source_results) and os.listdir(source_results):
        print(
            f"[Orchestrator] Migrating historical metrics to backup location: {backup_dir}"
        )
        os.makedirs(backup_dir, exist_ok=True)
        for file_name in os.listdir(source_results):
            source_file = os.path.join(source_results, file_name)
            if os.path.isfile(source_file):
                shutil.move(source_file, os.path.join(backup_dir, file_name))

    if os.path.exists(source_results):
        shutil.rmtree(source_results)
    os.makedirs(source_results, exist_ok=True)
    print("[Orchestrator] Telemetry space cleared. Starting with a fresh slate.")


def update_config(queue_type, alloc_type):
    """Updates config.json dynamically, handling thread counts and batch sizes
    to properly map onto the C++ ConfigManager processing cascade logic.
    """
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            try:
                config_data = json.load(f)
            except json.JSONDecodeError:
                config_data = {}
    else:
        config_data = {}

    if "engine" not in config_data or not isinstance(config_data["engine"], dict):
        config_data["engine"] = {}

    # Normalize values to lowercase to match C++ string comparison lookups
    config_data["engine"]["queue_type"] = queue_type.lower()
    config_data["engine"]["allocator_type"] = alloc_type.lower()

    # --- DYNAMIC BATCH CONFIGURATION ---
    # According to C++ logic: any queue string other than spsc/mpmc falls back to
    # the batch check. If batch_size > 1, it upgrades to QueueType::BATCHED.
    if queue_type == "BATCHED":
        config_data["engine"]["batch_size"] = (
            4  # Set greater than 1 to initialize BATCHED mode
        )
        print(
            "[Orchestrator] BATCHED Mode: Injecting batch_size = 4 into configurations."
        )
    else:
        config_data["engine"]["batch_size"] = (
            1  # Reset to 1 for conventional individual passes
        )
        print("[Orchestrator] Sequential Ingestion: Resetting batch_size = 1.")

    # --- DYNAMIC THREAD TOPOLOGY SCALE ---
    # SPSC is structurally restricted to a single consumer worker thread
    if queue_type == "SPSC":
        config_data["engine"]["thread_count"] = 1
        print(
            "[Orchestrator] SPSC Detected: Scaling server pool down to 1 isolated worker pod."
        )
    else:
        config_data["engine"]["thread_count"] = 8
        print(
            "[Orchestrator] Concurrency Enabled: Scaling server pool up to 8 worker pods."
        )

    config_data["engine"]["telemetry_csv_path"] = (
        f"../results/micro/squeezenet_{queue_type.lower()}_{alloc_type.lower()}.csv"
    )

    with open(CONFIG_PATH, "w") as f:
        json.dump(config_data, f, indent=4)


def run_matrix():
    cleanup_and_backup_environment()
    os.makedirs("results/micro", exist_ok=True)
    os.makedirs("results/macro", exist_ok=True)

    with open(os.devnull, "w") as devnull:
        for queue in QUEUE_TYPES:
            for alloc in ALLOCATOR_TYPES:
                print("\n=========================================================")
                print(f"[Orchestrator] EXECUTING MATRIX PAIRS: {queue} + {alloc}")
                print("=========================================================")

                update_config(queue, alloc)

                print(
                    "[Orchestrator] Launching server process silently inside build/ directory..."
                )
                server_proc = subprocess.Popen(
                    [SERVER_BINARY],
                    cwd="build",
                    stdout=devnull,
                    stderr=devnull,
                )

                time.sleep(1.5)

                if server_proc.poll() is not None:
                    print(
                        f"[Orchestrator] Error: Server failed to start for {queue} + {alloc}."
                    )
                    continue

                print(
                    "[Orchestrator] Launching client stress tests inside tests/client/ directory..."
                )

                # --- DYNAMIC CLIENT SCALE CONTROLLER ---
                client_cmd = ["python3", CLIENT_SCRIPT, queue, alloc]
                if queue == "SPSC":
                    client_cmd.append(
                        "--sequential"
                    )  # Instruct client to run with 1 thread

                try:
                    subprocess.run(
                        client_cmd,
                        cwd="tests/client",
                        check=True,
                    )
                except subprocess.CalledProcessError as e:
                    print(f"[Orchestrator] Error during client workload execution: {e}")

                print("[Orchestrator] Terminating server process...")
                try:
                    server_proc.send_signal(signal.SIGINT)
                    server_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    server_proc.kill()
                    server_proc.wait()

                print(f"[Orchestrator] Finished configuration pass: {queue} + {alloc}")

                try:
                    subprocess.run(
                        ["killall", "-9", "lumen_engine"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                except Exception:
                    pass
                time.sleep(1.0)

    print("\n[Orchestrator] Full macro and micro benchmarking matrix run complete!")


if __name__ == "__main__":
    run_matrix()
