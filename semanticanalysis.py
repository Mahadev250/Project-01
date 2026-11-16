
# import the libraries

import numpy as np
import pandas as pd
import re
import nltk
import emoji
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB

nltk.download('stopwords')
from nltk.corpus import stopwords

# load the dataset

df = pd.read_csv('semantic_dataset.csv', encoding='latin-1')

df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# Convert sentiment values (4 → 1 for positive)

df['target'] = df['target'].replace(4, 1);

# Keep only required columns

df = df[['target', 'text']]

# Stopwords

stop_words = set(stopwords.words('english'))

# Text cleaning function

def clean_text(text):

    # Convert emojis to words
    text = emoji.demojize(text)   # 😊 → :smiling_face:

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z_: ]", "", text)

    # Convert emoji text to readable words
    text = text.replace(":", " ")
    text = text.replace("_", " ")

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)

# Apply cleaning

df['clean_text'] = df['text'].apply(clean_text)

# WordCloud Visualization

text_data = " ".join(df['clean_text'])

wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white'
).generate(text_data)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("WordCloud of Tweets")
plt.show()

# Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'],
    df['target'],
    test_size=0.2,
    random_state=42
)

# TF-IDF Feature extraction
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Naive Bayes Model
nb_model = MultinomialNB()

nb_model.fit(X_train_vec, y_train)

nb_pred = nb_model.predict(X_test_vec)

nb_accuracy = accuracy_score(y_test, nb_pred)

print("Naive Bayes Accuracy:", nb_accuracy)


# Model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train_vec, y_train)

# Prediction
pred = model.predict(X_test_vec)

lr_accuracy = accuracy_score(y_test, pred)

models = ['Naive Bayes', 'Logistic Regression']
accuracies = [nb_accuracy, lr_accuracy]

plt.figure(figsize=(6,4))
plt.bar(models, accuracies)

plt.title("Model Comparison")
plt.ylabel("Accuracy")
plt.ylim(0,1)

plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative','Positive'],
            yticklabels=['Negative','Positive'])

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix")
plt.show()

# Accuracy Visualization
accuracy_value = accuracy_score(y_test, pred)

plt.figure(figsize=(5,4))
plt.bar(['Accuracy'], [accuracy_value])
plt.ylim(0,1)
plt.title("Model Accuracy")
plt.ylabel("Accuracy Score")
plt.show()

# Sentiment Distribution
sns.countplot(x=df['target'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Accuracy
accuracy = accuracy_score(y_test, pred)

print("Accuracy:", accuracy)

print("\nClassification Report\n")
print(classification_report(y_test, pred))