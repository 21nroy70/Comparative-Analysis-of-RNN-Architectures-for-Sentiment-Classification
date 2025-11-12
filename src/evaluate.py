"""
Evaluate & summarize results, make plots.

Usage examples:


1. Sorted tables + default seq-length plot (LSTM/RELU/ADAM, clipped=False)
python -m src.evaluate

2. Also make clipped=True seq-length figure
python -m src.evaluate --also_clipped true

3. Make the 3 optimal activation plots and print the picked rows
python -m src.evaluate --make_optimal_plots true

4. Alternate sort views
python -m src.evaluate --sort_by f1_desc
python -m src.evaluate --sort_by accuracy_desc
python -m src.evaluate --sort_by time_asc

"""

import os, argparse, shutil
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"
PLOTS_DIR   = os.path.join(RESULTS_DIR, "plots")
METRICS_CSV = os.path.join(RESULTS_DIR, "metrics.csv")

def read_metrics():
    if not os.path.exists(METRICS_CSV):
        raise FileNotFoundError(f"Could not find {METRICS_CSV}. Run train.py first.")
    df = pd.read_csv(METRICS_CSV)
    expected = ["Model","Activation","Optimizer","Seq Length","Grad Clipping","Accuracy","F1","Epoch Time (s)"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"metrics.csv missing columns: {missing}")
    return df

def sort_df(df, sort_by):
    if sort_by == "default":
        return df.sort_values(["Model","Activation","Optimizer","Seq Length", "Grad Clipping"])
    key = sort_by.lower()
    if key == "accuracy_desc":
        return df.sort_values("Accuracy", ascending=False)
    if key == "f1_desc":
        return df.sort_values("F1", ascending=False)
    if key == "time_asc":
        return df.sort_values("Epoch Time (s)", ascending=True)
    return df

def save_tables(df_sorted):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, "summary_table_sorted.csv")
    md_path  = os.path.join(RESULTS_DIR, "summary_table.md")
    df_sorted.to_csv(csv_path, index=False)
    with open(md_path, "w") as f:
        f.write(df_sorted.to_markdown(index=False))
    print(f"Saved table CSV → {csv_path}")
    print(f"Saved table MD  → {md_path}\n")

def print_table(df_sorted):
    print("==== SUMMARY TABLE (sorted) ====\n")
    print(df_sorted.to_string(index=False))
    print("\n================================\n")

def seq_length_plot(df, arch, activation, optimizer, clipped_flag):
    subset = df[
        (df["Model"].str.upper()      == arch.upper()) &
        (df["Activation"].str.upper() == activation.upper()) &
        (df["Optimizer"].str.upper()  == optimizer.upper()) &
        (df["Grad Clipping"].astype(str).str.lower() == ("yes" if clipped_flag else "no"))
    ].sort_values("Seq Length")

    if subset.empty:
        print(f"[seq] No rows for {arch}|{activation}|{optimizer}|clip={clipped_flag}")
        return None

    xs = subset["Seq Length"].tolist()
    acc = subset["Accuracy"].tolist()
    f1  = subset["F1"].tolist()
    mean_time = float(subset["Epoch Time (s)"].mean())

    os.makedirs(PLOTS_DIR, exist_ok=True)
    plt.figure()
    plt.plot(xs, acc, marker="o", label="Accuracy")
    plt.plot(xs, f1,  marker="o", label="Macro F1")
    plt.xlabel("Sequence Length"); plt.ylabel("Score")
    # include mean epoch time in title
    plt.title(
        f"Accuracy / F1 vs. Sequence Length\n"
        f"{arch.upper()} | {activation.upper()} | {optimizer.upper()} | "
        f"clipped={clipped_flag} | mean epoch time ≈ {mean_time:.2f}s"
    )
    plt.legend()
    out = os.path.join(PLOTS_DIR, f"seq_length_{arch}_{activation}_{optimizer}_clip{'True' if clipped_flag else 'False'}.png")
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"Saved: {out}")
    return out

def copy_loss_plot(src_name, dst_name):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    src = os.path.join(PLOTS_DIR, src_name)
    dst = os.path.join(PLOTS_DIR, dst_name)
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        print(f"Saved: {dst} (copied from {src_name})")
    else:
        print(f"[optimal] source not found: {src_name}")

def pick_row(df, model, act, opt, L, clip_yes=True):
    """Return a 1-row DataFrame for the requested configuration."""
    sub = df[
        (df["Model"].str.upper()==model.upper()) &
        (df["Activation"].str.upper()==act.upper()) &
        (df["Optimizer"].str.upper()==opt.upper()) &
        (df["Seq Length"]==int(L)) &
        (df["Grad Clipping"].astype(str).str.lower()==("yes" if clip_yes else "no"))
    ]
    return sub.head(1)

def print_and_save_optimal_summary(rows):
    """Print a compact table for the three optimal choices and save CSV/MD."""
    if not rows:
        print("[optimal] No rows to summarize.")
        return
    opt_df = pd.concat(rows, ignore_index=True)
    # Reorder columns for readability
    cols = ["Model","Activation","Optimizer","Seq Length","Grad Clipping","Accuracy","F1","Epoch Time (s)"]
    opt_df = opt_df[cols]
    print("==== OPTIMAL PICKS (by activation) ====\n")
    print(opt_df.to_string(index=False))
    print("\n=======================================\n")
    out_csv = os.path.join(RESULTS_DIR, "optimal_summary.csv")
    out_md  = os.path.join(RESULTS_DIR, "optimal_summary.md")
    opt_df.to_csv(out_csv, index=False)
    with open(out_md, "w") as f:
        f.write(opt_df.to_markdown(index=False))
    print(f"Saved optimal summary CSV → {out_csv}")
    print(f"Saved optimal summary MD  → {out_md}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sort_by", default="default",
                    help="default | accuracy_desc | f1_desc | time_asc")
    ap.add_argument("--arch", default="lstm",
                    help="For seq-length plots; e.g., lstm or bilstm")
    ap.add_argument("--activation", default="relu",
                    help="For seq-length plots; e.g., relu/tanh/sigmoid")
    ap.add_argument("--optimizer", default="adam",
                    help="For seq-length plots; e.g., adam/sgd/rmsprop")
    ap.add_argument("--also_clipped", default="false",
                    help="true/false: also generate clipped=True seq-length figure")
    ap.add_argument("--make_optimal_plots", default="false",
                    help="true/false: create your three specified 'optimal' plots and print their rows")
    args = ap.parse_args()

    df = read_metrics()
    df_sorted = sort_df(df, args.sort_by)
    save_tables(df_sorted)
    print_table(df_sorted)

    # seq-length plot(s)
    seq_length_plot(df, args.arch, args.activation, args.optimizer, clipped_flag=False)
    if str(args.also_clipped).lower() in ("true", "1", "yes"):
        seq_length_plot(df, args.arch, args.activation, args.optimizer, clipped_flag=True)

    # Optional: optimal plots + mini summary
    if str(args.make_optimal_plots).lower() in ("true", "1", "yes"):
        # copy plots
        copy_loss_plot("loss_lstm_relu_adam_L100_clipTrue.png",    "loss_optimal_relu.png")
        copy_loss_plot("loss_lstm_sigmoid_adam_L50_clipTrue.png",  "loss_optimal_sigmoid.png")
        copy_loss_plot("loss_lstm_tanh_adam_L50_clipTrue.png",     "loss_optimal_tanh.png")

        # print/highlight the exact rows you specified
        rows = []
        rows.append(pick_row(df_sorted, "lstm","relu","adam",   100, True))   # RELU
        rows.append(pick_row(df_sorted, "lstm","sigmoid","adam", 50, True))   # SIGMOID
        rows.append(pick_row(df_sorted, "lstm","tanh","adam",    50, True))   # TANH
        rows = [r for r in rows if not r.empty]
        print_and_save_optimal_summary(rows)

if __name__ == "__main__":
    main()
