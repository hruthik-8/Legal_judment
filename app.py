import os, json, pickle, torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from model_multitask import MultiTaskJudgmentModel

app = FastAPI()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Criminal model load
with open("checkpoints/model_config.json") as f:
    c_cfg = json.load(f)
c_tok = AutoTokenizer.from_pretrained(c_cfg["encoder"])
with open("checkpoints/encoders.pkl","rb") as f:
    encs = pickle.load(f)
le_charge = encs["le_charge"]
mlb_articles = encs["mlb_articles"]
criminal_model = MultiTaskJudgmentModel(c_cfg["encoder"], len(le_charge.classes_), len(mlb_articles.classes_)).to(DEVICE)
criminal_model.load_state_dict(torch.load("checkpoints/criminal-best.pt", map_location=DEVICE))
criminal_model.eval()

# Civil model load (if exists)
civil_model = None
if os.path.exists("checkpoints/civil.pt"):
    v_tok = AutoTokenizer.from_pretrained("checkpoints/civil-tokenizer")
    civil_model = torch.load("checkpoints/civil.pt", map_location=DEVICE)
    civil_model.eval()

class CriminalQuery(BaseModel):
    facts: str
    article_threshold: float = 0.5

class CivilQuery(BaseModel):
    facts: str
    plea: str
    law: str

@app.post("/predict_criminal")
def predict_criminal(q: CriminalQuery):
    enc = c_tok(q.facts, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = criminal_model(**enc)
        charge_idx = out["charge_logits"].argmax(dim=-1).item()
        charge = le_charge.inverse_transform([charge_idx])[0]
        art_scores = torch.sigmoid(out["articles_logits"]).squeeze(0).cpu().numpy().tolist()
        articles = []
        for i, score in enumerate(art_scores):
            if score >= q.article_threshold:
                articles.append({"article": mlb_articles.classes_[i], "score": float(score)})
        penalty = float(out["penalty_pred"].item())
    return {
        "charge": charge,
        "articles": sorted(articles, key=lambda x: -x["score"])[:10],
        "penalty_months": round(penalty,1)
    }

@app.post("/predict_civil")
def predict_civil(q: CivilQuery):
    if civil_model is None:
        return {"error": "Civil model not trained/loaded yet."}
    text = f"Facts: {q.facts}\nPlea: {q.plea}\nLaw: {q.law}"
    enc = v_tok(text, return_tensors="pt", truncation=True, max_length=384).to(DEVICE)
    with torch.no_grad():
        logit = civil_model(enc["input_ids"], enc["attention_mask"]).item()
    prob = float(torch.sigmoid(torch.tensor(logit)).item())
    return {
        "answer": "yes" if prob > 0.5 else "no",
        "probability": prob
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)