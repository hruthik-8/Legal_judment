import torch
from transformers import AutoTokenizer, AutoModel
from model_multitask import MultiTaskJudgmentModel
import json
import os
import torch.nn as nn

class CivilYesNoModel(nn.Module):
    def __init__(self, encoder_name: str, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        
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

# Load device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load civil model if available
if os.path.exists("checkpoints/civil-best.pt"):
    print("\n=== Loading Civil Model ===")
    try:
        # Load config
        with open("checkpoints/civil_config.json") as f:
            cfg = json.load(f)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg["encoder"])
        
        # Initialize and load model
        model = CivilYesNoModel(cfg["encoder"]).to(device)
        model.load_state_dict(torch.load("checkpoints/civil-best.pt", map_location=device))
        model.eval()
        
        # Test cases
        test_cases = [
            {
                "facts": "The plaintiff claims the defendant failed to repay a loan of ₹50,000 that was due on January 1st, 2023.",
                "plea": "The money was a gift, not a loan",
                "law": "Indian Contract Act"
            },
            {
                "facts": "The tenant has not paid rent for 6 months and the property is in good condition.",
                "plea": "I don't have the money right now",
                "law": "Rent Control Act"
            },
            {
                "facts": "The defendant signed a written loan agreement for ₹1,00,000 at 10% annual interest, with monthly payments of ₹10,000 starting January 2023. The defendant made no payments despite multiple written notices and the plaintiff has bank records showing the transfer of funds.",
                "plea": "I don't remember signing any agreement",
                "law": "Indian Contract Act"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            text = f"Facts: {test_case['facts']}\nPlea: {test_case['plea']}\nLaw: {test_case['law']}"
            inputs = tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=cfg["max_len"],
                padding="max_length"
            ).to(device)
            
            # Remove token_type_ids if present
            if 'token_type_ids' in inputs:
                del inputs['token_type_ids']
            
            with torch.no_grad():
                logit = model(inputs["input_ids"], inputs["attention_mask"]).item()
            
            prob = 1 / (1 + torch.exp(-torch.tensor(logit))).item()
            
            print(f"\nTest Case {i}:")
            print(f"Facts: {test_case['facts']}")
            print(f"Plea: {test_case['plea']}")
            print(f"Law: {test_case['law']}")
            print(f"Prediction: {'Yes' if prob > 0.5 else 'No'} (Confidence: {max(prob, 1-prob):.1%})")
            
    except Exception as e:
        print(f"\nError testing civil model: {str(e)}")
        raise e
else:
    print("\nCivil model not found at 'checkpoints/civil-best.pt'")