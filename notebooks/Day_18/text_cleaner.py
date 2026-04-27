import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

"""# Downloads 
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
"""
lemmatizer = WordNetLemmatizer()

def clean_bio(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # 3. Tokenize
    tokens = word_tokenize(text)

    # 4. Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in tokens if w not in stop_words]

    # 5. Lemmatization
    lemmatized = [lemmatizer.lemmatize(w) for w in filtered]

    return " ".join(lemmatized)


# Test
sample_bio = "I love Hiking in the mountains and Coding late at night!"
print("Cleaned Bio:", clean_bio(sample_bio))