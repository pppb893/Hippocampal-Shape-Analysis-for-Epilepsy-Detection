import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()

def main():
    parser = argparse.ArgumentParser(description="Plot ICP convergence time-series")
    parser.add_argument("--output_dir", type=str, help="Output directory containing icp_convergence_history.json")
    parser.add_argument("--show", action="store_true", help="Display the plot window")
    args, unknown = parser.parse_known_args()

    if args.output_dir:
        output_root = os.path.abspath(args.output_dir)
    else:
        output_root = os.path.join(SCRIPT_DIR, "output")

    json_path = os.path.join(output_root, "icp_convergence_history.json")
    output_plot = os.path.join(output_root, "icp_convergence.png")

    print(f"--- Plotting ICP Convergence Time-Series for {os.path.basename(output_root)} ---")

    if not os.path.exists(json_path):
        print(f"[ERROR] Convergence log not found: {json_path}")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    rounds = data.get("rounds", [])
    mean_dists = data.get("mean_distances", [])
    dist_changes = data.get("dist_changes", [])
    subject_dists = data.get("subject_distances", {})
    elapsed_times = data.get("elapsed_times_sec", [])

    pw_iters = data.get("pairwise_iterations", [])
    pw_dists = data.get("mean_pairwise_distances", [])
    subj_pw_dists = data.get("subject_pairwise_distances", {})

    if not rounds or not mean_dists:
        print("[ERROR] Convergence log is empty.")
        return

    x_vals_gw = rounds
    x_label_gw = "Groupwise Round"

    has_pairwise = bool(pw_iters) and bool(pw_dists)
    fig, ax = plt.subplots(figsize=(7.5, 6))

    # -------------------------------------------------------------
    # 95% Confidence Interval Shaded Band & Group Mean Line
    # -------------------------------------------------------------
    target_dict = subj_pw_dists if (has_pairwise and subj_pw_dists) else subject_dists
    target_x = pw_iters if (has_pairwise and subj_pw_dists) else x_vals_gw
    target_xlabel = "VTK Pairwise Iteration" if (has_pairwise and subj_pw_dists) else "Groupwise Round"

    subjs = [item for item in target_dict.items() if len(item[1]) == len(target_x)]
    num_subjs = len(subjs)

    mean_overlay = pw_dists if (has_pairwise and subj_pw_dists) else mean_dists

    # 95% Population Band for Group Variation around the Mean Line (Mean ± 1.96 SD)
    if subjs:
        subj_matrix = np.array([dists for _, dists in subjs])
        std_arr = np.std(subj_matrix, axis=0, ddof=1)
        mean_arr = np.array(mean_overlay) if len(mean_overlay) == len(target_x) else np.mean(subj_matrix, axis=0)

        ci_lower = np.maximum(0, mean_arr - 1.96 * std_arr)
        ci_upper = mean_arr + 1.96 * std_arr

        ax.fill_between(target_x, ci_lower, ci_upper, color='royalblue', alpha=0.22, label='95% Population Band (Mean ± 1.96 SD)')

    # Group mean overlay (thick dashed line)
    ax.plot(target_x, mean_overlay, color='black', linestyle='--', marker='o', linewidth=3.2, markersize=6, label='Group Mean Distance')
    ax.set_xlabel(target_xlabel, fontsize=12, fontweight='bold')
    ax.set_ylabel('Pairwise ICP Distance to Template', fontsize=12, fontweight='bold')
    ax.set_title(f'ICP Convergence Across VTK Iterations ({num_subjs} Meshes)', fontsize=13, fontweight='bold', pad=15)
    ax.set_xticks(target_x)
    ax.tick_params(axis='both', labelsize=11)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper right', fontsize=9.5)

    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    print(f"[OK] Time-series convergence plot saved to: {output_plot}")

    if args.show:
        try:
            print("[INFO] Displaying convergence plot window...")
            plt.show()
        except Exception as e:
            print(f"[NOTE] Could not open matplotlib window: {e}")
            try:
                if sys.platform == "win32":
                    os.startfile(output_plot)
            except Exception:
                pass

if __name__ == "__main__":
    main()
