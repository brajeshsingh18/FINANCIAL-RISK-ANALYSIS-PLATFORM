from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import pandas as pd

model_name="ProsusAI/finbert"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForSequenceClassification.from_pretrained(model_name)
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=model.to(device)
model.eval()

labels = {0: "positive",1: "negative",2: "neutral"}

def predict_sentiment(row:pd.Series):
    text=row["processed_text"]
    if pd.isna(text) or not str(text).strip():
        return {
            "sentiment": "neutral",
            "confidence": 1.0,
            "positive": 0.0,
            "negative": 0.0,
            "neutral": 1.0,
            "sentiment_score":0
        }
    input=tokenizer(text,return_tensors="pt",truncation=True,max_length=512,padding=True)
    input={k:v.to(device) for k , v in input.items()}
    with torch.no_grad():
        output=model(**input)

    probabilities = F.softmax(output.logits, dim=1).cpu().numpy()[0]
    prediction = int(probabilities.argmax())
    label=labels[prediction]
    if label == "positive":
        score = float(probabilities[prediction])
    elif label == "negative":
        score = -float(probabilities[prediction])
    else:
        score = 0.0
    return {
        "sentiment": labels[prediction],
        "confidence": float(probabilities[prediction]),
        "positive": float(probabilities[0]),
        "negative": float(probabilities[1]),
        "neutral": float(probabilities[2]),
        "sentiment_score":score
    }

# text = "Apple reports record quarterly earnings."

# label, score = predict_sentiment(text)

# print(label)
# print(score)



    
