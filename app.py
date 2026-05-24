from flask import Flask, request, jsonify, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import json

app = Flask(__name__)

# ── Training dataset ──────────────────────────────────────────────────────────
SPAM_EMAILS = [
    "Congratulations! You've won a $1,000,000 prize. Click here to claim now!",
    "FREE pills! Lose weight fast! Guaranteed results! Order now and get 50% off!",
    "URGENT: Your bank account has been compromised. Send your details immediately.",
    "You have been selected as a lucky winner. Provide your bank account to receive funds.",
    "Make money fast! Work from home and earn $5000 per week. No experience needed!",
    "Nigerian prince needs your help to transfer $47 million. You'll get 30% commission.",
    "Hot singles in your area! Click here to meet them tonight. Free registration!",
    "BUY CHEAP VIAGRA ONLINE! No prescription needed! Lowest prices guaranteed!",
    "You are pre-approved for a $50,000 loan! Apply now, no credit check required!",
    "Win a FREE iPhone 15! Just complete this survey. Limited time offer!!!",
    "FINAL NOTICE: Your account will be suspended. Click here immediately to verify.",
    "Earn $500 daily by clicking ads. No investment required. Join now for free!",
    "Exclusive deal: 90% off luxury watches. Replica designer brands. Click here!",
    "Your PayPal account is limited. Verify your information to restore access now.",
    "Amazing investment opportunity! Double your money in 30 days. Guaranteed returns!",
    "CONGRATULATIONS! Your email won our monthly lottery. Claim your $10,000 prize!",
    "Free vacation! You and a guest can travel to the Bahamas. Reply with your details.",
    "Cure diabetes naturally! This secret remedy doctors don't want you to know!",
    "Make $1000 per day from home doing simple tasks. No skills required!",
    "Your computer has a virus! Call this toll-free number immediately for support!",
    "Cheap Rolex watches! 95% off original price! Limited stock available!",
    "Cash advance approved! Get $5000 in your account today. Click to accept!",
    "You qualify for debt relief! We'll settle your debts for pennies on the dollar!",
    "Special offer for you only! Buy now and get 3 products for the price of 1!",
    "Increase your credit score by 200 points in 30 days. Guaranteed or money back!",
    "FREE GIFT! You have been chosen. Click here to claim your exclusive reward!",
    "Urgent message from IRS. You owe back taxes. Pay immediately to avoid arrest.",
    "Earn passive income online! Secret system revealed! Click now before it's gone!",
    "Your social security number has been suspended. Call us immediately!",
    "Become a millionaire overnight! This one weird trick will change your life forever!",
]

HAM_EMAILS = [
    "Hi Sarah, just confirming our meeting tomorrow at 2 PM in Conference Room B.",
    "Please find attached the quarterly report. Let me know if you have any questions.",
    "Your Amazon order #123-456 has been shipped and will arrive by Thursday.",
    "Hey! Want to grab lunch this week? I heard a new place opened downtown.",
    "Reminder: Team standup is at 9 AM tomorrow. Please review the sprint board beforehand.",
    "Happy birthday! Hope you have a wonderful day filled with joy and celebration.",
    "Can you review my pull request when you get a chance? No rush, whenever works.",
    "The project deadline has been moved to next Friday. Please update your tasks.",
    "Thank you for attending our webinar. Here are the slides and resources we discussed.",
    "Your flight to New York on June 15 has been confirmed. Check-in opens 24 hours before.",
    "Good morning! Just checking in on the status of the client proposal. Any updates?",
    "Dinner tonight at 7? We could try that Italian place on 5th Street.",
    "I've reviewed your resume and would love to schedule a call to discuss the role.",
    "Please bring your laptop to the workshop. We'll be doing hands-on exercises.",
    "The library books you requested are now available for pickup at the front desk.",
    "Heads up: the office will be closed on Monday for the public holiday.",
    "Your subscription renewal is coming up next month. No action needed if you want to continue.",
    "Great work on the presentation today! The client was really impressed.",
    "Can we reschedule our call to Thursday? Something came up on my end.",
    "The package was delivered to your front door at 3:45 PM this afternoon.",
    "Hi, I'm reaching out from the engineering team regarding your support ticket #4892.",
    "Please complete the onboarding form before your first day. It should take 15 minutes.",
    "Looking forward to the conference next week! Will you be at the networking dinner?",
    "Your gym membership is up for renewal. Visit any branch to renew in person.",
    "The budget spreadsheet has been updated with last month's figures. Please review.",
    "We're hosting a team lunch on Friday to celebrate the product launch. RSVP by Wednesday.",
    "Your appointment with Dr. Patel is confirmed for May 30 at 10:30 AM.",
    "The code review is done. A few minor comments but overall looks great, nice work!",
    "Can you send me the login credentials for the staging server? Need to test a fix.",
    "Monthly newsletter: Here's what's happening at the company this month.",
]

# ── Model training ─────────────────────────────────────────────────────────────
texts = SPAM_EMAILS + HAM_EMAILS
labels = [1] * len(SPAM_EMAILS) + [0] * len(HAM_EMAILS)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),
        max_features=5000,
        sublinear_tf=True
    )),
    ('nb', MultinomialNB(alpha=0.1))
])

pipeline.fit(X_train, y_train)

# ── Evaluate on held-out test set ──────────────────────────────────────────────
y_pred = pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
MODEL_METRICS = {
    "accuracy":  round(accuracy_score(y_test, y_pred) * 100, 1),
    "precision": round(precision_score(y_test, y_pred, zero_division=0) * 100, 1),
    "recall":    round(recall_score(y_test, y_pred, zero_division=0) * 100, 1),
    "f1":        round(f1_score(y_test, y_pred, zero_division=0) * 100, 1),
    "confusion_matrix": cm.tolist(),
    "train_size": len(X_train),
    "test_size":  len(X_test),
}

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", metrics=MODEL_METRICS)


@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Prediction
    prob = pipeline.predict_proba([text])[0]  # [ham_prob, spam_prob]
    is_spam = bool(prob[1] > 0.5)
    spam_prob = round(float(prob[1]) * 100, 1)
    ham_prob  = round(float(prob[0]) * 100, 1)

    # Top TF-IDF features for this text
    tfidf_vec   = pipeline.named_steps['tfidf']
    nb_model    = pipeline.named_steps['nb']
    transformed = tfidf_vec.transform([text])
    feature_names = tfidf_vec.get_feature_names_out()
    scores      = transformed.toarray()[0]
    top_indices = scores.argsort()[::-1][:10]

    top_features = []
    for i in top_indices:
        if scores[i] > 0:
            top_features.append({
                "word":  feature_names[i],
                "score": round(float(scores[i]), 4),
                "spam_weight": round(float(nb_model.feature_log_prob_[1][i]), 3),
            })

    return jsonify({
        "is_spam":    is_spam,
        "spam_prob":  spam_prob,
        "ham_prob":   ham_prob,
        "label":      "SPAM" if is_spam else "HAM",
        "top_features": top_features,
        "word_count": len(text.split()),
        "char_count": len(text),
    })


@app.route("/metrics")
def metrics():
    return jsonify(MODEL_METRICS)


if __name__ == "__main__":
    print("✅  Model trained successfully!")
    print(f"📊  Accuracy: {MODEL_METRICS['accuracy']}%  |  F1: {MODEL_METRICS['f1']}%")
    print("🚀  Starting server at http://localhost:5000")
    app.run(debug=True, port=5000)
