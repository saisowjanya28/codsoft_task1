

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

path = r"C:\Users\91903\OneDrive\Desktop\INT\task-4\spam.csv"
df = pd.read_csv(path, encoding="latin-1")

df =df.rename(columns={"v1": "label", "v2": "message"})
df = df[["label", "message"]]
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

tf = TfidfVectorizer(stop_words="english", max_features=3000)
X = tf.fit_transform(df['message'])
y = df['label']

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Bayes": MultinomialNB(),
    "SVM": SVC(kernel="linear"),
    "Logistic": LogisticRegression()
}

results = {}

for name, model in models.items():
    model.fit(trainX, trainY)
    predictions = model.predict(testX)
    acc = accuracy_score(testY, predictions)
    results[name] = acc
    print(f"{name} Accuracy: {acc * 100:.2f}%")

best = max(results, key=results.get)
best_model = models[best]

df["Prediction"] = best_model.predict(X)
df["Prediction"] = df["Prediction"].map({0: "HAM", 1: "SPAM"})

print(f"Best Model: {best} (Accuracy: {results[best] * 100:.2f}%)")
print(df.head(10))