#!/usr/bin/env bash

# --- MATRIX CONFIGURATION FOR REPRODUCIBILITY ---
QUEUE_TAG="MPMC"
ALLOC_TAG="LUMEN_ARENA"
CONFIG_NAME="config.json"

PROJECT_ROOT="$(pwd)"
CONFIG_PATH="${PROJECT_ROOT}/${CONFIG_NAME}"
RESULTS_DIR="${PROJECT_ROOT}/results/hardware_profile"

# --- SYSTEM PERMISSION CHECK ---
initialize_kernel_perf() {
    echo "[Profiler] Verifying kernel diagnostic capability..."
    local paranoid_level
    paranoid_level=$(cat /proc/sys/kernel/perf_event_paranoid)

    if [ "$paranoid_level" -gt 1 ]; then
        echo "[Profiler] Warning: Current perf_event_paranoid level is ${paranoid_level}."
        echo "[Profiler] Requesting temporary root privilege to lower restriction level to 1..."
        sudo sysctl -w kernel.perf_event_paranoid=1
    fi
    
    mkdir -p "${RESULTS_DIR}"
}

# --- SANITATION LOOP ---
clear_env() {
    killall -9 lumen_engine 2>/dev/null
    pkill -9 -f stress_test.py 2>/dev/null
    # Cooldown window to completely clear loopback TCP descriptors
    sleep 4.0
}

# --- PROFILING ENGINE UTILITY ---
execute_perf_sweep() {
    local concurrency=$1
    local output_log="${RESULTS_DIR}/perf_${concurrency}_clients.log"

    echo ""
    echo "========================================================="
    echo "[Profiler] STARTING HARDWARE TRACE: ${concurrency} CLIENTS"
    echo "========================================================="

    clear_env

    # 1. Update the static configuration parameters to guarantee a clean server state
    sed -i 's/"queue_type": "[^"]*"/"queue_type": "mpmc"/' "$CONFIG_PATH"
    sed -i 's/"allocator_type": "[^"]*"/"allocator_type": "lumen_arena"/' "$CONFIG_PATH"
    sed -i 's/"batch_size": [0-9]*/"batch_size": 1/' "$CONFIG_PATH"
    sed -i 's/"thread_count": [0-9]*/"thread_count": 8/' "$CONFIG_PATH"
    sed -i "s|\"telemetry_csv_path\": \"[^\"]*\"|\"telemetry_csv_path\": \"../results/micro/squeezenet_perf_${concurrency}.csv\"|" "$CONFIG_PATH"

    # 2. Launch the server process safely inside the /build directory
    echo "[Profiler] Launching server process inside build/..."
    cd "${PROJECT_ROOT}/build" || exit 1
    setsid ./lumen_engine >/dev/null 2>&1 &
    local server_pid=$!
    
    # Allow ONNX Runtime graph allocation states to reach equilibrium
    sleep 2.0

    if ! kill -0 $server_pid 2>/dev/null; then
        echo "[Profiler] Error: Engine failed to stabilize under PID ${server_pid}."
        cd "${PROJECT_ROOT}" || exit 1
        return 1
    fi

    # 3. Initialize the perf profiler attached to the target server execution process
    echo "[Profiler] Attaching performance counter monitoring to process ${server_pid}..."
    perf stat -p "$server_pid" \
      -e cycles,instructions,context-switches,cpu-migrations,page-faults \
      -e L1-dcache-loads,L1-dcache-load-misses \
      -e LLC-loads,LLC-load-misses \
      -o "$output_log" -- sleep 10 &
    local perf_pid=$!

    # 4. Jump to the client testing directory to maintain relative asset path lookups
    echo "[Profiler] Executing target workload generator inside tests/client/..."
    cd "${PROJECT_ROOT}/tests/client" || exit 1
    
    # Execute the client using setsid to isolate script context
    setsid -w python3 stress_test.py "$QUEUE_TAG" "$ALLOC_TAG"

    # 5. Await the full conclusion of the 10-second hardware measurement window
    echo "[Profiler] Syncing hardware profiling counters..."
    wait $perf_pid 2>/dev/null

    # 6. Tear down the engine process cleanly
    echo "[Profiler] Sending graceful SIGINT to server..."
    kill -2 $server_pid 2>/dev/null
    
    for i in {1..5}; do
        if ! kill -0 $server_pid 2>/dev/null; then
            break
        fi
        sleep 1
    done

    if kill -0 $server_pid 2>/dev/null; then
        kill -9 $server_pid 2>/dev/null
    fi
    wait $server_pid 2>/dev/null

    echo "[Profiler] Hardware log completely written to: ${output_log}"
    cd "${PROJECT_ROOT}" || exit 1
}

# --- MAIN AUTOMATION STREAM ---
initialize_kernel_perf

# Run Pass 1: Configure the client tool parameters for low concurrency
echo "[Profiler] Standardizing stress_test script to optimal load (4 Clients)..."
sed -i 's/NUM_CLIENTS = [0-9]*/NUM_CLIENTS = 4/' "${PROJECT_ROOT}/tests/client/stress_test.py"
sed -i 's/REQUESTS_PER_CLIENT = [0-9]*/REQUESTS_PER_CLIENT = 200/' "${PROJECT_ROOT}/tests/client/stress_test.py"
execute_perf_sweep 4

# Run Pass 2: Configure the client tool parameters for high contention saturation
echo "[Profiler] Standardizing stress_test script to oversaturated load (16 Clients)..."
sed -i 's/NUM_CLIENTS = [0-9]*/NUM_CLIENTS = 16/' "${PROJECT_ROOT}/tests/client/stress_test.py"
sed -i 's/REQUESTS_PER_CLIENT = [0-9]*/REQUESTS_PER_CLIENT = 50/' "${PROJECT_ROOT}/tests/client/stress_test.py"
execute_perf_sweep 16

# --- PRINT FINAL HARWARE DIAGNOSTICS RECAP ---
clear_env
echo ""
echo "========================================================="
echo "[Profiler] PIPELINE COMPLETE. SIDE-BY-SIDE ANALYTICS:"
echo "========================================================="
echo "Format: OPTIMAL (4 CLIENTS)             | SATURATED (16 CLIENTS)"
echo "---------------------------------------------------------"
diff -y "${RESULTS_DIR}/perf_4_clients.log" "${RESULTS_DIR}/perf_16_clients.log"
