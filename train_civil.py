import os, argparse, pickle, json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class CivilYesNoModel(torch.nn.Module):
    def __init__(self, encoder_name: str, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = torch.nn.Dropout(dropout)
        self.head = torch.nn.Linear(hidden, 1)  # sigmoid

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            h = out.pooler_output
        else:
            last_hidden = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1)
            h = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        h = self.dropout(h)
        return self.head(h).squeeze(-1)

class CivilDataset(Dataset):
    def __init__(self, df, tok, max_len=384):
        self.df = df.reset_index(drop=True)
        self.tok = tok; self.max_len = max_len

    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        text = f"Facts: {row['facts_text']}\nPlea: {row['pleas_text']}\nLaw: {row['law_context']}"
        enc = self.tok(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
        enc = {k:v.squeeze(0) for k,v in enc.items()}
        y = 1.0 if str(row["answer_text"]).strip().lower()=="yes" else 0.0
        enc["label"] = torch.tensor(y, dtype=torch.float32)
        return enc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="data/indian_kanoon_civil_cases.csv")
    ap.add_argument("--encoder", type=str, default="nlpaueb/legal-bert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=384)
    ap.add_argument("--outdir", type=str, default="checkpoints")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.csv)
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df   = train_test_split(temp_df, test_size=0.5, random_state=42)

    tok = AutoTokenizer.from_pretrained(args.encoder)
    train_ds = CivilDataset(train_df, tok, args.max_len)
    val_ds   = CivilDataset(val_df, tok, args.max_len)

    tr_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    va_loader = DataLoader(val_ds, batch_size=args.batch)

    model = CivilYesNoModel(args.encoder).to(DEVICE)
    opt = AdamW(model.parameters(), lr=args.lr)
    total_steps = len(tr_loader)*args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.1*total_steps), total_steps)

    bce = torch.nn.BCEWithLogitsLoss()

    def run_epoch(loader, train=True):
        model.train(train)
        tot=0.0
        for batch in tqdm(loader):
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            y = batch["label"].to(DEVICE)
            with torch.set_grad_enabled(train):
                logit = model(ids, mask)
                loss = bce(logit, y)
                if train:
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step(); sched.step()
            tot += float(loss.item())
        return tot/len(loader)

    best=1e9
    for ep in range(1, args.epochs+1):
        tr = run_epoch(tr_loader, True)
        va = run_epoch(va_loader, False)
        print(f"Epoch {ep} | train {tr:.4f} | valid {va:.4f}")
        torch.save(model.state_dict(), os.path.join(args.outdir, f"civil-ep{ep}.pt"))
        if va < best:
            best=va; torch.save(model.state_dict(), os.path.join(args.outdir, "civil-best.pt"))

    with open(os.path.join(args.outdir, "civil_config.json"), "w") as f:
        json.dump({"encoder": args.encoder, "max_len": args.max_len}, f)

if __name__ == "__main__":
    main()
