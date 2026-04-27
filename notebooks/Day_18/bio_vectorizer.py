from sklearn.feature_extraction.text import TfidfVectorizer

# Sample bios
bios = [
    "Expert in Python and Machine Learning for social good.",
    "Professional Chef who loves outdoor Hiking and mountains.",
    "Machine Learning enthusiast and mountain hiker."
]

# 1. Initialize
vectorizer = TfidfVectorizer()

# 2. Convert text → numbers
tfidf_matrix = vectorizer.fit_transform(bios)

# 3. Print results
print("Vocabulary:", vectorizer.get_feature_names_out())
print("Matrix Shape:", tfidf_matrix.toarray().shape)
print("TF-IDF Matrix:\n", tfidf_matrix.toarray())