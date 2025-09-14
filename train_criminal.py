import os, json, math, random, argparse, pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from model_multitask import MultiTaskJudgmentModel

def prepare_data(df):
    # Ensure lists
    df["charges_list"] = df["charges"].astype(str).str.split("|")
    df["articles_list"] = df["law_articles"].astype(str).str.split("|")
    # Penalty may be missing -> fill 0 and mask later
    df["penalty_months"] = pd.to_numeric(df["penalty_months"], errors="coerce").fillna(0).astype(int)
    return df

class JudgDataset(Dataset):
    def __init__(self, df, tok, le_charge, mlb_articles, max_len=512):
        self.df = df.reset_index(drop=True)
        self.tok = tok
        self.le_charge = le_charge
        self.mlb_articles = mlb_articles
        self.max_len = max_len

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        enc = self.tok(
            str(row["facts_text"]),
            truncation=True, padding="max_length", max_length=self.max_len,
            return_tensors="pt"
        )
        enc = {k:v.squeeze(0) for k,v in enc.items()}
        # charge: pick primary (first) for multi-class; also keep multi-hot if needed
        primary_charge = row["charges_list"][0] if row["charges_list"] else "Other"
        y_charge = self.le_charge.transform([primary_charge])[0]
        y_articles = self.mlb_articles.transform([row["articles_list"]])[0].astype(np.float32)
        y_penalty = float(row["penalty_months"])
        return {
            **enc,
            "y_charge": torch.tensor(y_charge, dtype=torch.long),
            "y_articles": torch.tensor(y_articles, dtype=torch.float32),
            "y_penalty": torch.tensor(y_penalty, dtype=torch.float32),
            "has_penalty": torch.tensor(1.0 if row["penalty_months"]>0 else 0.0, dtype=torch.float32)
        }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='data/indian_kanoon_criminal_cases.csv',
                        help='Path to criminal cases CSV')
    ap.add_argument("--encoder", type=str, default="nlpaueb/legal-bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--outdir", type=str, default="checkpoints")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.csv)
    df = prepare_data(df)

    # Encoders
    all_primary = [c[0] if isinstance(c, list) and len(c)>0 else "Other" for c in df["charges_list"]]
    le_charge = LabelEncoder().fit(all_primary + ["Other"])
    all_articles = sorted({a for lst in df["articles_list"] for a in (lst if isinstance(lst, list) else [])})
    mlb_articles = MultiLabelBinarizer(classes=all_articles)
    mlb_articles.fit([all_articles])

    # Save encoders
    with open(os.path.join(args.outdir, "encoders.pkl"), "wb") as f:
        pickle.dump({"le_charge": le_charge, "mlb_articles": mlb_articles, "articles_list": all_articles}, f)

    # Split
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

    tok = AutoTokenizer.from_pretrained(args.encoder)

    train_ds = JudgDataset(train_df, tok, le_charge, mlb_articles, args.max_len)
    val_ds   = JudgDataset(val_df, tok, le_charge, mlb_articles, args.max_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch)

    model = MultiTaskJudgmentModel(args.encoder, num_charges=len(le_charge.classes_), num_articles=len(mlb_articles.classes_)).to(DEVICE)
    opt = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(train_loader)*args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.1*total_steps), total_steps)

    ce = torch.nn.CrossEntropyLoss()
    bce = torch.nn.BCEWithLogitsLoss()
    mse = torch.nn.MSELoss()
    w_charge, w_articles, w_penalty = 1.0, 1.0, 0.1

    def run_epoch(loader, train=True):
        model.train(train)
        total=0.0
        for batch in tqdm(loader):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            y_charge = batch["y_charge"].to(DEVICE)
            y_articles = batch["y_articles"].to(DEVICE)
            y_penalty = batch["y_penalty"].to(DEVICE)
            has_pen = batch["has_penalty"].to(DEVICE)

            with torch.set_grad_enabled(train):
                out = model(input_ids, attention_mask)
                loss_charge = ce(out["charge_logits"], y_charge)
                loss_articles = bce(out["articles_logits"], y_articles)
                # mask penalty loss when label==0 (unknown)
                loss_penalty = (mse(out["penalty_pred"], y_penalty) * has_pen).sum() / (has_pen.sum().clamp(min=1.0))
                loss = w_charge*loss_charge + w_articles*loss_articles + w_penalty*loss_penalty
                if train:
                    opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
            total += float(loss.item())
        return total/len(loader)

    best = 1e9
    for ep in range(1, args.epochs+1):
        tr = run_epoch(train_loader, True)
        va = run_epoch(val_loader, False)
        print(f"Epoch {ep} | train {tr:.4f} | valid {va:.4f}")
        torch.save(model.state_dict(), os.path.join(args.outdir, f"criminal-ep{ep}.pt"))
        if va < best:
            best = va
            torch.save(model.state_dict(), os.path.join(args.outdir, "criminal-best.pt"))

    # Save tokenizer name for inference
    with open(os.path.join(args.outdir, "model_config.json"), "w") as f:
        json.dump({"encoder": args.encoder, "max_len": args.max_len}, f)

if __name__ == "__main__":
    main()
