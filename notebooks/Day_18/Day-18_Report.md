# Day 18 Report – Natural Language Processing (NLP)

## Objective

The goal of this task was to work with unstructured text data and build a basic NLP pipeline to clean, process, and analyze user-generated content. This included text preprocessing, feature extraction using TF-IDF, and sentiment analysis.

---

## 1. Text Preprocessing (NLP Cleaning Pipeline)

A Python script (`text_cleaner.py`) was implemented to clean raw text data.

### Steps performed:

* Converted text to lowercase
* Removed punctuation
* Tokenized text into words
* Removed stopwords (common words like "the", "is", etc.)
* Applied lemmatization to reduce words to their base form

### Output:

The cleaned text successfully removed noise and retained meaningful keywords.

---

## 2. Feature Extraction using TF-IDF

A second script (`bio_vectorizer.py`) was used to convert text data into numerical form using TF-IDF (Term Frequency-Inverse Document Frequency).

### Key Points:

* TF-IDF assigns importance to words based on frequency and uniqueness
* Rare but meaningful words receive higher scores
* Output is a numerical matrix representing text data

### Result:

The model successfully generated a vocabulary and corresponding TF-IDF matrix.

---

## 3. Sentiment Analysis

Sentiment analysis was performed using the TextBlob library on multiple sample user bios.

### Metrics:

* **Polarity**: Measures sentiment (-1 = negative, +1 = positive)
* **Subjectivity**: Measures opinion vs fact (0 = objective, 1 = subjective)

### Observations:

* Positive sentences showed high positive polarity
* Negative sentences showed strong negative polarity
* Mixed sentences resulted in near-neutral polarity

---

## 4. Stemming vs Lemmatization

* **Stemming**: Reduces words by cutting (e.g., "running" → "runn"), may not produce valid words
* **Lemmatization**: Converts words to meaningful root form (e.g., "running" → "run")

Lemmatization is more accurate and was used in this task.

---

## 5. Reflection

MeetMux can use sentiment analysis on event reviews to automatically identify negative user experiences. If multiple users provide low polarity scores for a particular event or organizer, the system can flag them for poor performance. This enables automated quality monitoring and helps maintain a better user experience on the platform.

---

## Conclusion

This task demonstrated how unstructured text data can be transformed into meaningful insights using NLP techniques. The pipeline successfully cleaned text, extracted features, and analyzed sentiment, forming a strong foundation for real-world AI applications.
