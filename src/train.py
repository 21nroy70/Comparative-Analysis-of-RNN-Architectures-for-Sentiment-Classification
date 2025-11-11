'''
How to run/execute this it (make sure you ran preprocess.py first to build vocab and clean data, 
                            you can see how to run it at the top comment of the program in preprocess.py):

                            


BTW, running all of these commands below (numbers 1 - 8, with 3-4 commands in each took FOREVER LOLLLLL so beaware - probably will take an hour or 2 but took me much longer bc of debugging and fixing and making sure it worked)

There's also like hundreds of unique ways to varying all factors to literally get each type of combo so I limited it down to like 1/5 of them, and even that took hours to run so have fun :) jk, just look at metrics.csv and run evaluate.py to see the summary and plots since it's already contained here when I ran this file to get the data/metrics

For a quick and the way I ran to debug and test out my code (it was hell lol) was running this below since it was quickiest:
python -m src.train --arch lstm --activation relu --optimizer adam --seq_len 50 --epochs 3

Now, let's actually do some stuff and tune up the hyperparemters and make it more scalable by being better and more complex:


Here is the hella experiements and variations across the different paremters to hypertune these models:

1. No clipping but changing Architecture: 

python -m src.train --arch rnn --activation relu --optimizer adam --seq_len 50 --epochs 6 

python -m src.train --arch lstm --activation relu --optimizer adam --seq_len 50 --epochs 6 

python -m src.train --arch bilstm --activation relu --optimizer adam --seq_len 50 --epochs 6 


Note: Alright, bilistm has the lowest loss over each epoch, but takes more than double the time than lstm with the exact same configs of optimizer, sequence length, etc. So, RNN is trash but will vary paremters of lstm and bilstm to see how they compare and find the optimal...


2. No clipping but changing Sequence Length: 

python -m src.train --arch lstm --activation relu --optimizer adam --seq_len 25 --epochs 6 

python -m src.train --arch lstm --activation relu --optimizer adam --seq_len 100 --epochs 6 

# BiLSTM
python -m src.train --arch bilstm --activation relu --optimizer adam --seq_len 25  --epochs 6
python -m src.train --arch bilstm --activation relu --optimizer adam --seq_len 100 --epochs 6



3. No clipping but changing Activation Functions: 

python -m src.train --arch lstm --activation tanh --optimizer adam --seq_len 50 --epochs 6 

python -m src.train --arch lstm --activation sigmoid --optimizer adam --seq_len 50 --epochs 6

# BiLISTM
python -m src.train --arch bilstm --activation tanh    --optimizer adam --seq_len 50 --epochs 6
python -m src.train --arch bilstm --activation sigmoid --optimizer adam --seq_len 50 --epochs 6



4. No clipping but changing Optimizers: 

python -m src.train --arch lstm --activation relu --optimizer sgd --seq_len 50 --epochs 6 --lr 0.01

python -m src.train --arch lstm --activation relu --optimizer rmsprop --seq_len 50 --epochs 6 

# BiLISTM
python -m src.train --arch bilstm --activation relu --optimizer sgd --seq_len 50 --epochs 6 --lr 0.01
python -m src.train --arch bilstm --activation relu --optimizer rmsprop --seq_len 50 --epochs 6



5. Clipping and changing Architecture: 
python -m src.train --arch rnn --activation relu --optimizer adam --seq_len 50 --epochs 6 --grad_clip 1.0 

python -m src.train --arch lstm --activation relu --optimizer adam --seq_len 50 --epochs 6 --grad_clip 1.0 

python -m src.train --arch bilstm --activation relu --optimizer adam --seq_len 50 --epochs 6 --grad_clip 1.0 



6. Clipping and changing Sequence Length: 

python -m src.train --arch lstm --activation relu --optimizer adam --seq_len 25 --epochs 6 --grad_clip 1.0 

python -m src.train --arch lstm --activation relu --optimizer adam --seq_len 100 --epochs 6 --grad_clip 1.0 

# BiLSTM with gradient clipping
python -m src.train --arch bilstm --activation relu --optimizer adam --seq_len 25  --epochs 6 --grad_clip 1.0
python -m src.train --arch bilstm --activation relu --optimizer adam --seq_len 100 --epochs 6 --grad_clip 1.0


7. Clipping and changing Activation Functions: 

python -m src.train --arch lstm --activation tanh --optimizer adam --seq_len 50 --epochs 6 --grad_clip 1.0 

python -m src.train --arch lstm --activation sigmoid --optimizer adam --seq_len 50 --epochs 6 --grad_clip 1.0 

#BiLISTM with gradient clipping
python -m src.train --arch bilstm --activation tanh --optimizer adam --seq_len 50 --epochs 6 --grad_clip 1.0
python -m src.train --arch bilstm --activation sigmoid --optimizer adam --seq_len 50 --epochs 6 --grad_clip 1.0



8. Clipping and changing Optimizers: 

python -m src.train --arch lstm --activation relu --optimizer sgd --seq_len 50 --epochs 6 --grad_clip 1.0 --lr 0.01

python -m src.train --arch lstm --activation relu --optimizer rmsprop --seq_len 50 --epochs 6 --grad_clip 1.0

# BiLISTM with clipping
python -m src.train --arch bilstm --activation relu --optimizer sgd     --seq_len 50 --epochs 6 --grad_clip 1.0 --lr 0.01
python -m src.train --arch bilstm --activation relu --optimizer rmsprop --seq_len 50 --epochs 6 --grad_clip 1.0





'''




import os, argparse, time, json, pickle, csv
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, SGD, RMSprop
import hashlib


# Adding other 2 program files I write with their functions for this 
from .utils import seed_everything, make_loader, compute_metrics, load_json
from .models import RNNClassifier


# allowed optimizers
OPTIMIZERS = {"adam": Adam, "sgd": SGD, "rmsprop": RMSprop}

def load_examples(processed_dir: str):
    """Load encoded examples + vocab produced by preprocess.py."""
    with open(os.path.join(processed_dir, "train.pkl"), "rb") as f:
        train_ex = pickle.load(f)
    with open(os.path.join(processed_dir, "test.pkl"), "rb") as f:
        test_ex = pickle.load(f)
    vocab = load_json(os.path.join(processed_dir, "vocab.json"))
    return train_ex, test_ex, vocab

def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=None):
    """Standard train loop over one epoch; optional gradient clipping."""
    model.train()
    losses = []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        p = model(x)                 # probability in [0,1]
        loss = criterion(p, y)       # BCE loss
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))

@torch.no_grad()
def evaluate(model, loader, device):
    """Compute Accuracy and macro-F1 on a dataloader."""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        ys.append(y.numpy())
        ps.append(model(x).detach().cpu().numpy())
    y_true = np.concatenate(ys)
    y_prob = np.concatenate(ps)
    acc, f1 = compute_metrics(y_true, y_prob)
    return acc, f1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--results_dir",   default="results")

    # assignment-controlled factors
    parser.add_argument("--arch", choices=["rnn", "lstm", "bilstm"], default="lstm")
    parser.add_argument("--activation", choices=["relu", "tanh", "sigmoid"], default="relu")
    parser.add_argument("--optimizer", choices=["adam", "sgd", "rmsprop"], default="adam")
    parser.add_argument("--seq_len", type=int, choices=[25, 50, 100], default=50)
    parser.add_argument("--grad_clip", type=float, default=None, help="e.g., 1.0 to enable clipping")

    # fixed hyperparams per spec (exposed just in case)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr",     type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout",    type=float, default=0.5)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    seed_everything(42)
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(args.results_dir, "plots"), exist_ok=True)



    # data
    train_ex, test_ex, vocab = load_examples(args.processed_dir)
    train_loader = make_loader(train_ex, args.seq_len, args.batch_size, shuffle=True)
    test_loader  = make_loader(test_ex,  args.seq_len, args.batch_size, shuffle=False)

    # model
    model = RNNClassifier(
        vocab_size=len(vocab),
        emb_dim=100,
        hidden_size=args.hidden_size,
        num_layers=2,
        arch=args.arch,
        dropout=args.dropout,
        head_activation=args.activation
    ).to(args.device)

    # optimizer + loss
    opt_cls = OPTIMIZERS[args.optimizer]
    optimizer = opt_cls(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # training
    losses, epoch_times = [], []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, args.device, grad_clip=args.grad_clip)
        dt = time.time() - t0
        losses.append(tr_loss)
        epoch_times.append(dt)
        print(f"Epoch {epoch}/{args.epochs} — loss={tr_loss:.4f} — {dt:.1f}s")

    # evaluation
    acc, f1 = evaluate(model, test_loader, args.device)
    mean_time = float(np.mean(epoch_times))

    # append to metrics.csv (summary table for the report)
    metrics_path = os.path.join(args.results_dir, "metrics.csv")
    write_header = not os.path.exists(metrics_path)
    with open(metrics_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["Model", "Activation", "Optimizer", "Seq Length",
                        "Grad Clipping", "Accuracy", "F1", "Epoch Time (s)"])
        w.writerow([
            args.arch.upper(), args.activation.upper(), args.optimizer.upper(),
            args.seq_len, "Yes" if args.grad_clip is not None else "No",
            round(acc, 4), round(f1, 4), round(mean_time, 2)
        ])

    # training loss plot — we’ll use these later for best/worst models in the report
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(range(1, len(losses) + 1), losses)
    plt.xlabel("Epoch"); plt.ylabel("Training Loss")
    plt.title(f"{args.arch.upper()} | {args.activation.upper()} | {args.optimizer.upper()} | L={args.seq_len} | clip={args.grad_clip}")
    plot_path = os.path.join(args.results_dir, "plots",
                             f"loss_{args.arch}_{args.activation}_{args.optimizer}_L{args.seq_len}_clip{bool(args.grad_clip)}.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    # single-run JSON snapshot (hhelped me debug to know what the metrics looked like for my last run - was so painful to have to delete the .csv file each time so decided to add this)
    with open(os.path.join(args.results_dir, "last_run.json"), "w") as f:
        json.dump({
            "model": args.arch,
            "activation": args.activation,
            "optimizer": args.optimizer,
            "seq_len": args.seq_len,
            "grad_clipping": args.grad_clip is not None,
            "accuracy": float(acc),
            "macro_f1": float(f1),
            "mean_epoch_time_sec": mean_time,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "emb_dim": 100
        }, f, indent=2)

    print(f" Eval — acc={acc:.4f}, f1={f1:.4f}, mean epoch time={mean_time:.2f}s")
    print(" metrics.csv updated at:", metrics_path)

if __name__ == "__main__":
    main()