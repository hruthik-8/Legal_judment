import torch
from transformers import AutoTokenizer
from model_multitask import MultiTaskJudgmentModel
import json
import os

# Load the model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load config and tokenizer
with open("checkpoints/model_config.json") as f:
    cfg = json.load(f)
tokenizer = AutoTokenizer.from_pretrained(cfg["encoder"])

# Load encoders
with open("checkpoints/encoders.pkl", "rb") as f:
    import pickle
    encs = pickle.load(f)
le_charge = encs["le_charge"]
mlb_articles = encs["mlb_articles"]

# Initialize and load model
model = MultiTaskJudgmentModel(
    cfg["encoder"], 
    num_charges=len(le_charge.classes_), 
    num_articles=len(mlb_articles.classes_)
).to(device)
model.load_state_dict(torch.load("checkpoints/criminal-best.pt", map_location=device))
model.eval()

# Test prediction
# facts = "The defendant was caught stealing a bicycle from a shop."
facts = "The accused assaulted a police officer during a protest."
article_threshold = 0.3

# Tokenize
inputs = tokenizer(facts, return_tensors="pt", truncation=True, max_length=512).to(device)

# Remove token_type_ids if present
if 'token_type_ids' in inputs:
    del inputs['token_type_ids']

# Predict
with torch.no_grad():
    outputs = model(**inputs)

# Process outputs
charge_idx = outputs["charge_logits"].argmax(dim=-1).item()
charge = le_charge.inverse_transform([charge_idx])[0]

article_scores = torch.sigmoid(outputs["articles_logits"]).squeeze(0).cpu().numpy()
articles = []
for i, score in enumerate(article_scores):
    if score >= article_threshold:
        articles.append({"article": mlb_articles.classes_[i], "score": float(score)})

penalty = float(outputs["penalty_pred"].item())

# Print results
print("\n=== Criminal Prediction ===")
print(f"Facts: {facts}")
print(f"\nMost likely charge: {charge}")
print("\nRelevant articles:")
for art in sorted(articles, key=lambda x: -x["score"])[:5]:  # Top 5 articles
    print(f"- {art['article']}: {art['score']:.2f}")
print(f"\nEstimated penalty: {penalty:.1f} months")