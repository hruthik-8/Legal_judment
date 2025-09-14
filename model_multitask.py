import torch
import torch.nn as nn
from transformers import AutoModel

class MultiTaskJudgmentModel(nn.Module):
    def __init__(self, encoder_name: str, num_charges: int, num_articles: int, dropout=0.2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.charge_head = nn.Linear(hidden, num_charges)     # softmax later
        self.articles_head = nn.Linear(hidden, num_articles)  # sigmoid later
        self.penalty_head = nn.Linear(hidden, 1)              # regression

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            h = out.pooler_output
        else:
            last_hidden = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1)
            h = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        h = self.dropout(h)
        return {
            "charge_logits": self.charge_head(h),
            "articles_logits": self.articles_head(h),
            "penalty_pred": self.penalty_head(h).squeeze(-1),
            "hidden": h
        }
