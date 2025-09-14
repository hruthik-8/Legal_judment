import pandas as pd
df = pd.read_csv("data/indian_kanoon_civil_cases.csv")
print(df["answer_text"].value_counts())
print("\nSample 'Yes' cases:")
print(df[df["answer_text"].str.strip().str.lower()=="yes"][["facts_text", "pleas_text", "law_context"]].head(2).to_string())