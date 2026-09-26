"""
GenAI Lab: Sentiment Analysis using a Pre-trained Hugging Face Model
----------------------------------------------------------------------
Real-world application:
An e-commerce platform (e.g., Amazon/Flipkart) wants to automatically
gauge customer satisfaction from product review text, without manual
reading of thousands of reviews. This helps the business flag negative
reviews for quick support follow-up and track overall product sentiment.

Steps:
1. Install/import Hugging Face 'transformers' library
2. Load the sentiment-analysis pipeline (pre-trained DistilBERT model
   fine-tuned on SST-2)
3. Feed sample customer review sentences and print sentiment + score
"""

# Step 0: Install (run once in Colab/terminal, not needed if already installed)
# !pip install transformers torch

from transformers import pipeline

# Step 1: Load the pre-trained sentiment-analysis pipeline
# By default this uses distilbert-base-uncased-finetuned-sst-2-english
sentiment_analyzer = pipeline("sentiment-analysis")

# Step 2: Sample real-world input — customer product reviews
reviews = [
    "This phone's battery life is amazing, I'm extremely happy with it!",
    "The delivery was late and the packaging was completely damaged.",
    "Decent quality for the price, does the job.",
    "Worst purchase I've ever made, it stopped working in two days.",
    "Absolutely loved the fabric and stitching, will buy again!"
]

# Step 3: Run sentiment analysis on each review
print("=== Customer Review Sentiment Analysis ===\n")
for review in reviews:
    result = sentiment_analyzer(review)[0]
    label = result["label"]
    score = round(result["score"] * 100, 2)
    print(f"Review   : {review}")
    print(f"Sentiment: {label} (confidence: {score}%)\n")

# Step 4 (optional): Summarize overall sentiment distribution
positive = sum(1 for r in reviews if sentiment_analyzer(r)[0]["label"] == "POSITIVE")
negative = len(reviews) - positive
print("=== Summary ===")
print(f"Positive reviews: {positive}")
print(f"Negative reviews: {negative}")
