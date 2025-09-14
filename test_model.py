import torch
from transformers import AutoTokenizer
from model_multitask import MultiTaskJudgmentModel

def test_model():
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskJudgmentModel("bert-base-uncased", num_charges=10, num_articles=20).to(device)
    
    # Test input
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text = "The defendant was caught stealing a bicycle."
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    
    # Forward pass - remove token_type_ids if present
    inputs.pop('token_type_ids', None)
    with torch.no_grad():
        outputs = model(**inputs)
    
    print("Model outputs:")
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")

if __name__ == "__main__":
    test_model()
