from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_data(df):
    X = df['message']
    y = df['label']

    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=3000
    )
    X_vectorized = vectorizer.fit_transform(X)

    return X_vectorized, y, vectorizer
