#!/usr/bin/env bash

# --- MATRIX DEFINITIONS ---
QUEUE_TYPES=("naive_mutex" "spsc" "mpmc" "batched")
ALLOCATOR_TYPES=("standard" "lumen_arena")

CONFIG_NAME="config.json"

# Capture the absolute project root directory path
PROJECT_ROOT="$(pwd)"
CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_NAME}"

# Absolute sanitation loop for sequential passes
clear_env() {
    killall -9 lumen_engine 2>/dev/null
    pkill -9 -f stress_test.py 2>/dev/null
    
    # CRITICAL: Cooldown window allowing the Linux kernel network layer 
    # to fully flush out lingering TCP TIME_WAIT state descriptors.
    sleep 4.0
}

# Global Environment Setup (Port of cleanup_and_backup_environment)
initialize_results_directory() {
    echo "[Orchestrator] Running system process sanitation step..."
    killall -9 lumen_engine 2>/dev/null
    pkill -9 -f stress_test.py 2>/dev/null
    sleep 1.0

    local backup_dir="${PROJECT_ROOT}/../lumen_telemetry_backup"
    local source_results="${PROJECT_ROOT}/results"

    # Check if the results directory exists and is not empty
    if [ -d "$source_results" ] && [ "$(ls -A "$source_results" 2>/dev/null)" ]; then
        echo "[Orchestrator] Migrating historical metrics to backup location: $backup_dir"
        mkdir -p "$backup_dir"
        
        # Move all files and folders to the backup location
        mv "$source_results"/* "$backup_dir/" 2>/dev/null
    fi

    # Wipe and recreate completely clean target folders
    rm -rf "$source_results"
    mkdir -p "${source_results}/micro" "${source_results}/macro"
    echo "[Orchestrator] Telemetry space cleared. Starting with a fresh slate."
}

# Run the global pipeline initialization once before executing the loops
initialize_results_directory

for queue in "${QUEUE_TYPES[@]}"; do
    for alloc in "${ALLOCATOR_TYPES[@]}"; do
        echo ""
        echo "========================================================="
        echo "[Orchestrator] EXECUTING MATRIX PAIRS: ${queue^^} + ${alloc^^}"
        echo "========================================================="
        
        clear_env

        # State configurations
        batch_size=1
        thread_count=8
        client_flags=""

        if [ "$queue" == "batched" ]; then
            batch_size=4
            echo "[Orchestrator] BATCHED Mode: Injecting batch_size = 4 into configurations."
        else
            echo "[Orchestrator] Sequential Ingestion: Resetting batch_size = 1."
        fi

        if [ "$queue" == "spsc" ]; then
            thread_count=1
            client_flags="--sequential"
            echo "[Orchestrator] SPSC Detected: Scaling server pool down to 1 isolated worker pod."
        else
            echo "[Orchestrator] Concurrency Enabled: Scaling server pool up to 8 worker pods."
        fi

        # Stream modification of config.json using absolute path reference
        sed -i 's/"queue_type": "[^"]*"/"queue_type": "'"$queue"'"/' "$CONFIG_PATH"
        sed -i 's/"allocator_type": "[^"]*"/"allocator_type": "'"$alloc"'"/' "$CONFIG_PATH"
        sed -i 's/"batch_size": [0-9]*/"batch_size": '"$batch_size"'/' "$CONFIG_PATH"
        sed -i 's/"thread_count": [0-9]*/"thread_count": '"$thread_count"'/' "$CONFIG_PATH"
        sed -i 's|"telemetry_csv_path": "[^"]*"|"telemetry_csv_path": "../results/micro/squeezenet_'"$queue"'_'"$alloc"'.csv"|' "$CONFIG_PATH"

        echo "[Orchestrator] Launching server process inside build/ via detached Session..."
        cd "${PROJECT_ROOT}/build" || exit 1
        
        # PERFECTED SOLUTION: setsid handles full logical scheduling group detachment.
        # No physical pinning is applied, allowing ONNX and worker threads to load balance dynamically.
        setsid ./lumen_engine >/dev/null 2>&1 &
        SERVER_PID=$!
        
        # Let server initialize core resources and ONNX structures cleanly
        sleep 2.0

        # Check if server stayed up cleanly before kicking off workload engines
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "[Orchestrator] Error: Server failed to start for ${queue^^} + ${alloc^^}."
            cd "${PROJECT_ROOT}" || exit 1
            continue
        fi

        echo "[Orchestrator] Launching client stress tests inside tests/client/..."
        cd "${PROJECT_ROOT}/tests/client" || exit 1
        
        # Run client workload generator natively without physical restrictions
        setsid -w python3 stress_test.py "${queue^^}" "${alloc^^}" $client_flags

        echo "[Orchestrator] Terminating server process cleanly..."
        kill -2 $SERVER_PID 2>/dev/null
        
        # Wait for clean SIGINT handling, force kill if it hangs longer than 5 seconds
        for i in {1..5}; do
            if ! kill -0 $SERVER_PID 2>/dev/null; then
                break
            fi
            sleep 1
        done
        
        if kill -0 $SERVER_PID 2>/dev/null; then
            kill -9 $SERVER_PID 2>/dev/null
        fi
        wait $SERVER_PID 2>/dev/null
        
        echo "[Orchestrator] Finished configuration pass: ${queue^^} + ${alloc^^}"
        
        # Return to project root for the next loop iteration
        cd "${PROJECT_ROOT}" || exit 1
    done
done

# Final system sanitation pass
killall -9 lumen_engine 2>/dev/null
echo ""
echo "[Orchestrator] Full macro and micro benchmarking matrix run complete with clean session isolation!"
