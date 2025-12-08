from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
emails = [
"Win money now",
"Lowest price for meds",
"Meeting at 10am tomorrow",
"Congratulations you won",
"Lunch with team today",
"Can you call me tomorrow?",
"Confirm your email",
]
labels = [1,1,0,1,0,0,1]
vectorizer = CountVectorizer()
X=vectorizer.fit_transform(emails)
model = LogisticRegression()
model.fit(X, labels)
test_emails = ["Win a free ticket now",
"Team meeting rescheduled",
"Cheap meds available here",
'You won 1000$! Congratulations',
"I'm sorry, I can't call right now",
"Confirm your email address",
'We have launch with team'
]
X_test = vectorizer.transform(test_emails)
predictions = model.predict(X_test)

for email, pred in zip(test_emails, predictions):
    print(f"Email: '{email}' -> Prediction: {'SPAM' if pred==1 else 'NOT SPAM'}")