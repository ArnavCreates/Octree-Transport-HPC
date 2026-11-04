#!/usr/bin/env python3
"""Run the seven solver regimes and build figures from measured telemetry."""

import json
import os
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
EXE_NAME = "octree_transport_hpc.exe" if os.name == "nt" else "octree_transport_hpc"
BIN_PATH = os.path.join(BUILD_DIR, EXE_NAME)
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_engine(version, n, steps=10, init=1, dt=0.005, softening=0.05, theta=0.5,
               integrator="verlet", threads=None, compare_direct=False,
               reset_after_warmup=False):
    handle = tempfile.NamedTemporaryFile(prefix="octree_telemetry_", suffix=".json", delete=False)
    telemetry_path = handle.name
    handle.close()
    cmd = [
        BIN_PATH,
        "--version", str(version),
        "-n", str(n),
        "--steps", str(steps),
        "--init", str(init),
        "--dt", str(dt),
        "--softening", str(softening),
        "--theta", str(theta),
        "--integrator", integrator,
        "--headless",
        "--json-out", telemetry_path,
    ]
    if compare_direct:
        cmd.append("--compare-direct")
    if reset_after_warmup:
        cmd.append("--reset-after-warmup")

    env = os.environ.copy()
    if threads is not None:
        env["OMP_NUM_THREADS"] = str(threads)
        env["OMP_DYNAMIC"] = "FALSE"

    try:
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Engine failed (v={version}, N={n}, exit={proc.returncode}).\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        with open(telemetry_path, "r", encoding="utf-8") as telemetry_file:
            return json.load(telemetry_file)
    finally:
        try:
            os.remove(telemetry_path)
        except OSError:
            pass


def measured_max_drift(result, name):
    return result.get(name, result.get("max_relative_energy_drift", 0.0))


def main():
    print("Octree solver measurements: scaling, force error, and energy drift")
    if not os.path.isfile(BIN_PATH):
        raise FileNotFoundError(f"Executable not found: {BIN_PATH}. Build the Release target first.")

    benchmark_data = {
        "metadata": {
            "platform": platform.platform(),
            "cpu_architecture": platform.machine(),
            "processor": platform.processor() or platform.machine(),
            "python_version": platform.python_version(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "energy_drift_sampling": "every timed step for N <= 16384; first and final timed steps for larger N",
        },
        "scaling_matrix": {},
        "thread_scaling": {},
        "theta_sweep": {},
        "symplectic_results": {},
        "close_approach_stress": {},
    }
    observed_renderers = []

    print("\n>>> Seven-regime solver scaling matrix")
    n_values_direct_seq = [1024, 4096, 16384]
    n_values_direct_omp = [1024, 4096, 16384, 32768]
    n_values_direct_gpu = [1024, 4096, 16384, 32768, 65536]
    n_values_bh_seq = [1024, 4096, 16384, 32768, 65536]
    n_values_bh_parallel = [1024, 4096, 16384, 32768, 65536, 131072]
    regimes = [
        (1, "v1_cpu_seq", n_values_direct_seq),
        (2, "v2_cpu_omp", n_values_direct_omp),
        (3, "v3_gpu_naive", n_values_direct_gpu),
        (4, "v4_gpu_tiled", n_values_direct_gpu),
        (5, "v5_bh_seq", n_values_bh_seq),
        (6, "v6_bh_omp", n_values_bh_parallel),
        (7, "v7_bh_gpu", n_values_bh_parallel),
    ]
    for version, name, sizes in regimes:
        values = {
            "N": [], "mean_step_ms": [], "mean_tree_ms": [], "mean_force_ms": [],
            "particles_per_sec": [], "max_energy_drift": [], "max_momentum_drift": [],
            "max_angular_momentum_drift": [],
        }
        for n in sizes:
            steps = 2 if n >= 65536 and version in (3, 4) else (
                3 if n >= 65536 and version == 5 else (5 if n >= 32768 else 10)
            )
            print(f"  {name}, N={n}, steps={steps} ...", end="", flush=True)
            result = run_engine(version, n, steps=steps, init=1)
            if version in (3, 4, 7):
                observed_renderers.append(result.get("renderer", ""))
            values["N"].append(n)
            values["mean_step_ms"].append(result["mean_step_ms"])
            values["mean_tree_ms"].append(result["mean_tree_ms"])
            values["mean_force_ms"].append(result["mean_force_ms"])
            values["particles_per_sec"].append(result["particles_per_sec"])
            values["max_energy_drift"].append(measured_max_drift(result, "max_energy_drift"))
            values["max_momentum_drift"].append(result["max_momentum_drift"])
            values["max_angular_momentum_drift"].append(result["max_angular_momentum_drift"])
            print(f" {result['mean_step_ms']:.3f} ms/step")
        benchmark_data["scaling_matrix"][name] = values

    print("\n>>> OpenMP thread scaling for v6")
    thread_results = {
        "threads": [], "mean_step_ms": [], "mean_tree_ms": [], "mean_force_ms": [], "speedup": [],
    }
    baseline_ms = None
    for thread_count in [1, 2, 4, 8, 12]:
        print(f"  threads={thread_count}, N=32768 ...", end="", flush=True)
        result = run_engine(6, 32768, steps=10, threads=thread_count)
        if thread_count == 1:
            baseline_ms = result["mean_step_ms"]
        thread_results["threads"].append(thread_count)
        thread_results["mean_step_ms"].append(result["mean_step_ms"])
        thread_results["mean_tree_ms"].append(result["mean_tree_ms"])
        thread_results["mean_force_ms"].append(result["mean_force_ms"])
        thread_results["speedup"].append(baseline_ms / result["mean_step_ms"])
        print(f" {result['mean_step_ms']:.3f} ms ({thread_results['speedup'][-1]:.2f}x)")
    benchmark_data["thread_scaling"] = thread_results

    print("\n>>> Measured Barnes-Hut theta sweep")
    theta_values = [0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.2]
    theta_results = {
        "theta": [], "mean_step_ms": [], "mean_force_ms": [],
        "relative_l2_force_error": [], "speedup_vs_direct": [],
    }
    for theta in theta_values:
        print(f"  theta={theta} ...", end="", flush=True)
        result = run_engine(6, 4096, steps=10, init=2, theta=theta,
                            compare_direct=True, reset_after_warmup=True)
        if result["relative_l2_force_error"] is None or result["speedup_vs_direct"] is None:
            raise RuntimeError(f"Missing direct-force comparison telemetry for theta={theta}")
        theta_results["theta"].append(theta)
        theta_results["mean_step_ms"].append(result["mean_step_ms"])
        theta_results["mean_force_ms"].append(result["mean_force_ms"])
        theta_results["relative_l2_force_error"].append(result["relative_l2_force_error"])
        theta_results["speedup_vs_direct"].append(result["speedup_vs_direct"])
        print(f" {result['relative_l2_force_error'] * 100:.4f}% error, {result['speedup_vs_direct']:.2f}x")
    benchmark_data["theta_sweep"] = theta_results

    print("\n>>> Euler and Velocity Verlet trajectory comparison")
    particle_count = 1024
    trajectory_steps = 1000
    trajectory_dt = 0.002
    trajectory_results = {}
    for key, label, version, integrator, theta in [
        ("euler", "Forward Euler (direct OpenMP)", 2, "euler", 0.5),
        ("verlet", "Velocity Verlet (direct OpenMP)", 2, "verlet", 0.5),
        ("verlet_barnes_hut", "Velocity Verlet (Barnes-Hut OpenMP)", 6, "verlet", 0.5),
    ]:
        print(f"  {label}, {trajectory_steps} steps ...", end="", flush=True)
        result = run_engine(version, particle_count, steps=trajectory_steps, init=7,
                            dt=trajectory_dt, theta=theta, integrator=integrator,
                            reset_after_warmup=True)
        trajectory_results[key] = result
        print(f" max energy drift={result['max_energy_drift']:.6e}")

    euler = trajectory_results["euler"]
    verlet = trajectory_results["verlet"]
    verlet_bh = trajectory_results["verlet_barnes_hut"]
    benchmark_data["symplectic_results"] = {
        "particles": particle_count,
        "steps": trajectory_steps,
        "dt": trajectory_dt,
        "euler_initial_energy": euler["initial_energy"],
        "euler_final_energy": euler["final_energy"],
        "euler_max_energy_drift": euler["max_energy_drift"],
        "euler_max_momentum_drift": euler["max_momentum_drift"],
        "euler_max_angular_momentum_drift": euler["max_angular_momentum_drift"],
        "verlet_initial_energy": verlet["initial_energy"],
        "verlet_final_energy": verlet["final_energy"],
        "verlet_max_energy_drift": verlet["max_energy_drift"],
        "verlet_max_momentum_drift": verlet["max_momentum_drift"],
        "verlet_max_angular_momentum_drift": verlet["max_angular_momentum_drift"],
        "verlet_barnes_hut_max_energy_drift": verlet_bh["max_energy_drift"],
        "verlet_barnes_hut_max_momentum_drift": verlet_bh["max_momentum_drift"],
        "verlet_barnes_hut_max_angular_momentum_drift": verlet_bh["max_angular_momentum_drift"],
        "drift_reduction_ratio": euler["max_energy_drift"] / max(verlet["max_energy_drift"], 1.0e-15),
    }

    print("\n>>> Close-approach tree measurement")
    stress_n = 4096
    stress_steps = 1
    stress_result = run_engine(6, stress_n, steps=stress_steps, init=8, dt=0.0,
                               reset_after_warmup=True)
    stress_histogram = stress_result["octree_depth_histogram"]
    if len(stress_histogram) != 21:
        raise RuntimeError(f"Expected 21 depth bins, received {len(stress_histogram)}")
    print(f"  nodes={stress_result['octree_nodes']}, max_depth={stress_result['octree_max_depth']}, "
          f"coincident_merges={stress_result['coincident_merges']}")
    benchmark_data["close_approach_stress"] = {
        "particles": stress_n,
        "timed_steps": stress_steps,
        "initializer": "CloseApproachStress",
        "coincident_pairs": 50,
        "ultra_close_pairs": 50,
        "dense_core_particles": 64,
        "octree_max_depth": stress_result["octree_max_depth"],
        "octree_nodes": stress_result["octree_nodes"],
        "octree_depth_histogram": stress_histogram,
        "coincident_merges": stress_result["coincident_merges"],
        "max_energy_drift": stress_result["max_energy_drift"],
        "max_momentum_drift": stress_result["max_momentum_drift"],
        "max_angular_momentum_drift": stress_result["max_angular_momentum_drift"],
    }

    benchmark_data["metadata"]["openmp_max_threads"] = max(
        (result.get("openmp_max_threads", 1) for result in trajectory_results.values()), default=1
    )
    benchmark_data["metadata"]["renderer"] = next(
        (renderer for renderer in observed_renderers if renderer and "Unknown" not in renderer),
        "Not reported by a GPU run",
    )

    json_path = os.path.join(ASSETS_DIR, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8", newline="\n") as output_file:
        json.dump(benchmark_data, output_file, indent=2)
        output_file.write("\n")
    print(f"\n[Saved] {json_path}")

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 11, "legend.fontsize": 9, "grid.alpha": 0.35})

    colors = {
        "v1_cpu_seq": "#e74c3c", "v2_cpu_omp": "#e67e22", "v3_gpu_naive": "#f1c40f",
        "v4_gpu_tiled": "#2ecc71", "v5_bh_seq": "#9b59b6", "v6_bh_omp": "#1abc9c",
        "v7_bh_gpu": "#2980b9",
    }
    labels = {
        "v1_cpu_seq": "v1 CPU direct sequential", "v2_cpu_omp": "v2 CPU direct OpenMP",
        "v3_gpu_naive": "v3 GPU direct naive", "v4_gpu_tiled": "v4 GPU direct tiled",
        "v5_bh_seq": "v5 Barnes-Hut sequential", "v6_bh_omp": "v6 Barnes-Hut OpenMP",
        "v7_bh_gpu": "v7 Barnes-Hut GLSL",
    }
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    for name, values in benchmark_data["scaling_matrix"].items():
        ax1.plot(values["N"], values["mean_step_ms"], marker="o", linewidth=1.8,
                 color=colors[name], label=labels[name])
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Particle count N")
    ax1.set_ylabel("Mean step latency (ms)")
    ax1.set_title("Seven solver regimes")
    ax1.legend(frameon=True, fontsize=8)
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    ax2.plot(thread_results["threads"], thread_results["speedup"], marker="o", linewidth=2.2,
             color="#1abc9c", label="Measured speedup")
    ax2.plot(thread_results["threads"], thread_results["threads"], linestyle="--",
             color="#7f8c8d", label="Ideal scaling")
    ax2.set_xlabel("OpenMP threads")
    ax2.set_ylabel("Speedup relative to one thread")
    ax2.set_title("v6 thread scaling, N=32768")
    ax2.legend(frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    plot1_path = os.path.join(ASSETS_DIR, "solver_scaling_7regimes.png")
    fig.savefig(plot1_path, dpi=300)
    plt.close(fig)
    print(f"[Saved] {plot1_path}")

    euler_history = np.asarray(euler["energy_history"], dtype=float)
    verlet_history = np.asarray(verlet["energy_history"], dtype=float)
    bh_history = np.asarray(verlet_bh["energy_history"], dtype=float)
    euler_drift = np.abs((euler_history - euler["initial_energy"]) / abs(euler["initial_energy"]))
    verlet_drift = np.abs((verlet_history - verlet["initial_energy"]) / abs(verlet["initial_energy"]))
    bh_drift = np.abs((bh_history - verlet_bh["initial_energy"]) / abs(verlet_bh["initial_energy"]))
    steps_axis = np.arange(1, trajectory_steps + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    ax1.plot(steps_axis, np.maximum(euler_drift, 1.0e-16), label="Forward Euler, direct OpenMP", color="#e74c3c")
    ax1.plot(steps_axis, np.maximum(verlet_drift, 1.0e-16), label="Velocity Verlet, direct OpenMP", color="#2ecc71")
    ax1.plot(steps_axis, np.maximum(bh_drift, 1.0e-16), label="Velocity Verlet, Barnes-Hut", color="#3498db", linestyle="--")
    ax1.set_yscale("log")
    ax1.set_xlabel("Timed step")
    ax1.set_ylabel("Relative mechanical energy drift")
    ax1.set_title("Energy conservation over 1000 steps")
    ax1.legend(frameon=True)
    ax1.grid(True, which="both", linestyle="--", alpha=0.4)
    ratio = euler_drift / np.maximum(verlet_drift, 1.0e-15)
    ax2.plot(steps_axis, np.maximum(ratio, 1.0e-15), color="#8e44ad", linewidth=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("Timed step")
    ax2.set_ylabel("Euler drift / Verlet drift")
    ax2.set_title(f"Drift reduction: {benchmark_data['symplectic_results']['drift_reduction_ratio']:.3g}x")
    ax2.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    plot2_path = os.path.join(ASSETS_DIR, "symplectic_energy_drift.png")
    fig.savefig(plot2_path, dpi=300)
    plt.close(fig)
    print(f"[Saved] {plot2_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    ax1_twin = ax1.twinx()
    line_error = ax1.plot(theta_results["theta"],
                          np.asarray(theta_results["relative_l2_force_error"]) * 100,
                          color="#c0392b", marker="s", linewidth=2.1,
                          label="Measured relative L2 error (%)")
    line_latency = ax1_twin.plot(theta_results["theta"], theta_results["mean_step_ms"],
                                 color="#2980b9", marker="o", linestyle="--", linewidth=2.1,
                                 label="Mean step latency (ms)")
    ax1.set_xlabel("Barnes-Hut opening angle theta")
    ax1.set_ylabel("Relative force L2 error (%)", color="#c0392b")
    ax1_twin.set_ylabel("Mean step latency (ms)", color="#2980b9")
    ax1.set_title("Measured accuracy and latency")
    ax1.legend(line_error + line_latency, [line.get_label() for line in line_error + line_latency], loc="best")
    histogram = np.asarray(stress_histogram, dtype=np.int64)
    depths = np.arange(len(histogram))
    ax2.bar(depths, histogram, color="#16a085", edgecolor="#0e6655", width=0.8)
    ax2.set_yscale("symlog", linthresh=1)
    ax2.set_xlabel("Octree depth")
    ax2.set_ylabel("Occupied node count")
    ax2.set_title(f"Measured close-approach tree ({benchmark_data['close_approach_stress']['coincident_merges']} merges)")
    ax2.set_xticks(np.arange(0, len(histogram), 2))
    ax2.grid(True, axis="y", which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()
    plot3_path = os.path.join(ASSETS_DIR, "close_approach_stress.png")
    fig.savefig(plot3_path, dpi=300)
    plt.close(fig)
    print(f"[Saved] {plot3_path}")

    print("\nBenchmark artifacts written")
    print(f" Results JSON: {json_path}")
    print(f" Figure 1:     {plot1_path}")
    print(f" Figure 2:     {plot2_path}")
    print(f" Figure 3:     {plot3_path}")


if __name__ == "__main__":
    main()
