"""
How to run this/execute on command line 
(make sure you executed the 3 commands I put in the readme as well as the top comments in the requirements.txt file
so you're also in the same anoconda area with correct python version to be able to use the requirments.txt versions ):

python -m src.preprocess --csv data/IMDB_Dataset.csv --out_dir data/processed --vocab_size 10000




Preprocess IMDb CSV:
- lowercase + regex-tokenize
- deterministic 50/50 split per class (train: 25k, test: 25k)
- vocab from TRAIN ONLY (top-10k incl. PAD/OOV)
- encode to id lists (unpadded; padding is chosen later per run)
Outputs: data/processed/{train.pkl,test.pkl,vocab.json,stats.json}
"""
import os, argparse, pickle, pandas as pd
from .utils import seed_everything, simple_tokenize, build_vocab, encode, save_json, SentimentExample

def stratified_half_split(df: pd.DataFrame):
    train_idx, test_idx = [], []
    for _, g in df.groupby("sentiment", sort=True):
        g = g.sort_index()
        half = len(g)//2
        train_idx += g.index[:half].tolist()
        test_idx  += g.index[half:].tolist()
    return df.loc[train_idx], df.loc[test_idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to IMDB_Dataset.csv")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--vocab_size", type=int, default=10000)
    args = ap.parse_args()

    seed_everything(42)
    os.makedirs(args.out_dir, exist_ok=True)

    # Read CSV (must have 'review' and 'sentiment' columns)
    df = pd.read_csv(args.csv)
    assert {"review","sentiment"}.issubset(df.columns)
    df["label"] = (df["sentiment"].str.lower() == "positive").astype(int)

    # Tokenize each review: lowercase, keep [a-z0-9]+
    df["tokens"] = df["review"].astype(str).apply(simple_tokenize)

    # Deterministic 50/50 split within each class
    train_df, test_df = stratified_half_split(df)

    # Build vocab from TRAIN ONLY (top 10k)
    vocab = build_vocab(train_df["tokens"].tolist(), max_size=args.vocab_size)

    # Encode tokens → ids (unpadded)
    def enc_frame(frame):
        return [SentimentExample(ids=encode(toks, vocab), label=int(lbl))
                for toks, lbl in zip(frame["tokens"], frame["label"])]

    train_ex = enc_frame(train_df); test_ex = enc_frame(test_df)

    # Persist artifacts for later steps
    with open(os.path.join(args.out_dir, "train.pkl"), "wb") as f: pickle.dump(train_ex, f)
    with open(os.path.join(args.out_dir, "test.pkl"), "wb") as f:  pickle.dump(test_ex, f)

    save_json(vocab, os.path.join(args.out_dir, "vocab.json"))
    stats = {
        "n_train": len(train_ex),
        "n_test": len(test_ex),
        "vocab_size": len(vocab),
        "avg_len_train": sum(len(e.ids) for e in train_ex)/len(train_ex),
        "avg_len_test": sum(len(e.ids) for e in test_ex)/len(test_ex),
    }
    save_json(stats, os.path.join(args.out_dir, "stats.json"))
    print("Preprocessing complete →", args.out_dir)
    print(stats)

if __name__ == "__main__":
    main()
