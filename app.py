# ===== app.py =====
import os
import re
import warnings
import torch
import pandas as pd
from difflib import SequenceMatcher
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    BlenderbotSmallTokenizer,
    BlenderbotSmallForConditionalGeneration
)

# ====== SETUP ======
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

# ====== Load Sentiment Model ======
print("Loading fine-tuned sentiment model...")
tokenizer = BertTokenizer.from_pretrained("./fine_tuned_sentiment_model")
model = BertForSequenceClassification.from_pretrained("./fine_tuned_sentiment_model")
id2label = {0: "negative", 1: "neutral", 2: "positive"}

# ====== Load BlenderBot ======
print("Loading BlenderBot...")
blender_tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot_small-90M")
blender_model = BlenderbotSmallForConditionalGeneration.from_pretrained("facebook/blenderbot_small-90M")

# ====== Load Custom Responses CSV ======
RESPONSES_FILE = "custom_responses.csv"
if not os.path.exists(RESPONSES_FILE):
    pd.DataFrame(columns=["emotion", "chat", "response"]).to_csv(RESPONSES_FILE, index=False)
custom_responses = pd.read_csv(RESPONSES_FILE).dropna()

# ====== Sentiment Detection ======
def detect_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = logits.argmax().item()
    return id2label[predicted_class_id]

# ====== BlenderBot Fallback ======
def generate_blender_response(user_input):
    inputs = blender_tokenizer([user_input], return_tensors="pt")
    reply_ids = blender_model.generate(
        **inputs,
        max_length=60,
        do_sample=True,
        top_k=30,
        top_p=0.9,
        temperature=0.6,
        repetition_penalty=1.2
    )
    return blender_tokenizer.decode(reply_ids[0], skip_special_tokens=True)

# ====== Find Custom Response ======
def find_custom_response(user_input, sentiment):
    def similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    filtered = custom_responses[custom_responses["emotion"] == sentiment]
    best_match = max(filtered.itertuples(), key=lambda row: similarity(user_input, row.chat), default=None)

    if best_match and similarity(user_input, best_match.chat) > 0.6:
        return best_match.response
    return None

# ====== Chat API ======
@app.route("/chat", methods=["POST"])
def chat_handler():
    user_input = request.json.get("text") or request.json.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Empty input"}), 400

    if re.search(r"\b(hi|hello|hey)\b", user_input, re.I):
        return jsonify({
            "response": "Hi there! How are you feeling today?",
            "sentiment": "neutral",
            "question": "How are you feeling today?"
        })

    sentiment = detect_sentiment(user_input)
    custom_response = find_custom_response(user_input, sentiment)

    final_response = custom_response if custom_response else generate_blender_response(user_input)

    return jsonify({
        "response": final_response,
        "sentiment": sentiment,
        "question": "Would you like to talk more about it?"
    })

# ====== Feedback Logging API ======
@app.route("/feedback", methods=["POST"])
def log_feedback():
    data = request.json
    user_input = data.get("user_input", "")
    bot_response = data.get("bot_response", "")
    sentiment = data.get("sentiment", "")
    feedback = data.get("feedback", "")

    with open("feedback_log.csv", "a") as f:
        f.write(f'"{user_input}","{bot_response}","{sentiment}","{feedback}"\n')

    return jsonify({"status": "Feedback logged"}), 200

# ====== Run App ======
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
