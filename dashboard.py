import glob
import os

import pandas as pd
import plotly.express as px
import streamlit as st

# Set page configuration
st.set_page_config(page_title="Lumen Engine Performance Dashboard", layout="wide")


# --- Data Loading ---
@st.cache_data
def load_data(directory="results"):
    """
    Loads all CSV files from the specified directory and combines them into a single DataFrame.
    """
    all_files = glob.glob(os.path.join(directory, "*.csv"))

    if not all_files:
        st.error(
            f"No CSV files found in directory: '{directory}'. Please check the path."
        )
        return pd.DataFrame()

    df_list = []
    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            # Ensure required columns exist
            required_columns = {
                "queue_type",
                "alloc_type",
                "total_ms",
                "inference_ms",
                "queue_ms",
            }
            if not required_columns.issubset(df.columns):
                st.warning(
                    f"Skipping {filename}: Missing columns {required_columns - set(df.columns)}"
                )
                continue
            df_list.append(df)
        except Exception as e:
            st.error(f"Error reading {filename}: {e}")

    if not df_list:
        return pd.DataFrame()

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df["Configuration"] = (
        combined_df["queue_type"] + " + " + combined_df["alloc_type"]
    )
    return combined_df


# --- Helper Functions ---
def calculate_throughput(df):
    if df.empty:
        return 0
    duration = df["timestamp"].max() - df["timestamp"].min()
    if duration <= 0:
        duration = 1
    return len(df) / duration


def calculate_p99(df):
    return df["total_ms"].quantile(0.99)


# --- Main Application ---
st.title("⚡ Lumen Inference Engine Benchmark Results")
st.markdown("""
This dashboard visualizes the performance metrics of the **Lumen Inference Engine**, comparing different 
**Concurrency Models** (MPMC, SPSC, Mutex) and **Memory Allocators** (Lumen Arena vs. Standard Malloc).
""")

# Load Data
data_dir = "results"
df = load_data(data_dir)

if not df.empty:
    # --- Sidebar Controls ---
    st.sidebar.header("Filter Configuration")

    all_queues = df["queue_type"].unique().tolist()
    all_allocs = df["alloc_type"].unique().tolist()

    selected_queues = st.sidebar.multiselect(
        "Select Queue Types", all_queues, default=all_queues
    )
    selected_allocs = st.sidebar.multiselect(
        "Select Allocator Types", all_allocs, default=all_allocs
    )

    # --- NEW: Chart Style Toggle ---
    st.sidebar.markdown("---")
    st.sidebar.header("Chart Settings")
    chart_mode = st.sidebar.radio(
        "Jitter Chart Mode", ["Separate (Faceted)", "Overlaid (Comparison)"], index=0
    )

    # Filter DataFrame
    filtered_df = df[
        (df["queue_type"].isin(selected_queues))
        & (df["alloc_type"].isin(selected_allocs))
    ]

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        # --- Metrics Summary ---
        st.header("🏆 Performance Summary")
        metrics_df = (
            filtered_df.groupby(["Configuration", "queue_type", "alloc_type"])
            .apply(
                lambda x: pd.Series(
                    {
                        "RPS": calculate_throughput(x),
                        "P99 Latency (ms)": calculate_p99(x),
                        "Avg Latency (ms)": x["total_ms"].mean(),
                        "Max Latency (ms)": x["total_ms"].max(),
                        "Request Count": len(x),
                    }
                )
            )
            .reset_index()
        )

        best_config = metrics_df.loc[metrics_df["P99 Latency (ms)"].idxmin()]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Lowest P99 Latency",
            f"{best_config['P99 Latency (ms)']:.2f} ms",
            best_config["Configuration"],
        )
        col2.metric("Highest Throughput", f"{metrics_df['RPS'].max():.0f} RPS")
        col3.metric("Total Requests Processed", f"{len(filtered_df):,}")
        col4.metric("Configs Compared", len(metrics_df))

        # --- Visualizations ---
        st.subheader("1. Latency Analysis")
        c1, c2 = st.columns(2)
        with c1:
            fig_p99 = px.bar(
                metrics_df.sort_values("P99 Latency (ms)"),
                x="Configuration",
                y="P99 Latency (ms)",
                color="alloc_type",
                text_auto=".2f",
                title="99th Percentile Latency by Configuration",
            )
            fig_p99.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_p99, use_container_width=True)
        with c2:
            fig_box = px.box(
                filtered_df,
                x="Configuration",
                y="total_ms",
                color="alloc_type",
                title="Latency Distribution",
            )
            fig_box.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("2. Throughput & Bottlenecks")
        c3, c4 = st.columns(2)
        with c3:
            fig_rps = px.bar(
                metrics_df.sort_values("RPS", ascending=False),
                x="Configuration",
                y="RPS",
                color="queue_type",
                text_auto=".0f",
                title="Requests Per Second",
            )
            fig_rps.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_rps, use_container_width=True)
        with c4:
            melted_df = filtered_df.melt(
                id_vars=["Configuration"],
                value_vars=[
                    "queue_ms",
                    "preprocess_ms",
                    "inference_ms",
                    "postprocess_ms",
                ],
                var_name="Stage",
                value_name="Time (ms)",
            )
            avg_components = (
                melted_df.groupby(["Configuration", "Stage"])["Time (ms)"]
                .mean()
                .reset_index()
            )
            fig_stack = px.bar(
                avg_components,
                x="Configuration",
                y="Time (ms)",
                color="Stage",
                title="Average Time Spent per Stage",
                barmode="stack",
            )
            fig_stack.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_stack, use_container_width=True)

        # --- 3. Stability Over Time (Dynamic) ---
        st.subheader("3. Stability Over Time")
        st.markdown(
            "Observe latency spikes or jitter over the duration of the benchmark run."
        )

        line_chart_data = filtered_df.sort_values("timestamp")
        if len(line_chart_data) > 5000:
            line_chart_data = line_chart_data.iloc[::5, :]

        if chart_mode == "Separate (Faceted)":
            num_configs = len(line_chart_data["Configuration"].unique())
            dynamic_height = max(400, 250 * num_configs)
            fig_line = px.line(
                line_chart_data,
                x="request_id",
                y="total_ms",
                color="Configuration",
                title="Latency Jitter (Separate Rows)",
                markers=True,
                facet_row="Configuration",
                height=dynamic_height,
            )
            fig_line.update_xaxes(matches=None, showticklabels=False, title=None)
            fig_line.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        else:
            # Overlaid Mode
            fig_line = px.line(
                line_chart_data,
                x="request_id",
                y="total_ms",
                color="Configuration",
                title="Latency Jitter (Overlaid Comparison)",
                markers=True,
                height=600,
            )
            fig_line.update_xaxes(title="Request Sequence ID")

        st.plotly_chart(fig_line, use_container_width=True)

        with st.expander("See Summary Data Table"):
            st.dataframe(
                metrics_df.style.highlight_min(
                    axis=0, subset=["P99 Latency (ms)"], color="lightgreen"
                )
            )

else:
    st.info("Please ensure your CSV files are in the 'results' folder.")
