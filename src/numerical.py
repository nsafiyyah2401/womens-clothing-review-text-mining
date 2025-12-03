# NUMERICAL REPRESENTATION: TFIDF & BoW
import pandas as pd
import nltk
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')

# Load the Excel file from your specified path
file_path = 'C:\\Users\\Safiyyah\\eclipse-workspace\\ASSIGN-NEW ECOMMERCE WOMAN CLOTHING.xlsx'
df = pd.read_excel(file_path)

# Get the list of stop words from NLTK and the English vocabulary
stop_words = set(stopwords.words("english"))
english_vocab = set(words.words())  # Set of valid English words

# Initialize the Porter Stemmer for stemming
porter = PorterStemmer()

# Define a function to clean and preprocess the text
def preprocess_text(text):
    if isinstance(text, str):  # Check if text is a string
        # Tokenize the text into individual words
        tokens = word_tokenize(text)
        # Convert all tokens to lowercase
        tokens = [word.lower() for word in tokens]
        # Remove stop words
        tokens = [word for word in tokens if word not in stop_words]
        # Remove non-alphabetic tokens
        tokens = [word for word in tokens if word.isalpha()]
        # Remove words that are not in the English vocabulary or are too short
        tokens = [word for word in tokens if word in english_vocab and len(word) > 2]
        # Stem each token to its root form
        #tokens = [porter.stem(word) for word in tokens]
        # Join the tokens back into a single string
        cleaned_text = " ".join(tokens)
        return cleaned_text
    else:
        return ""  # Return an empty string for non-string inputs

# Apply the preprocessing function to the 'REVIEW TEXT' column and store it in a new column 'CLEANED_TEXT'
df['CLEANED_TEXT'] = df['REVIEW TEXT'].apply(preprocess_text)

# Drop the original 'REVIEW TEXT' column
df = df.drop(columns=['REVIEW TEXT'])

# Save the cleaned data with all columns except the original 'REVIEW TEXT' to a new Excel file
output_file_path = 'C:\\Users\\Safiyyah\\eclipse-workspace\\processed_ASSIGN-ECOMMERCE WOMAN CLOTHING.xlsx'
df.to_excel(output_file_path, index=False)
print(f"Preprocessed Cleaned Text data successfully saved to {output_file_path}")

# Print only the cleaned text from the 'CLEANED_TEXT' column
print("\nCLEANED_TEXT:\n")
print(df['CLEANED_TEXT'].head(10))  # Display the first 10 rows of the 'CLEANED_TEXT' column

# Apply TF-IDF on the 'CLEANED_TEXT' column
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['CLEANED_TEXT'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=[f"D{i+1}" for i in range(len(df))], columns=tfidf_vectorizer.get_feature_names_out())

# Print the TF-IDF Matrix
print("\nTF-IDF Matrix:\n", tfidf_df)

# Save the TF-IDF matrix to CSV with words as columns and documents as rows
tfidf_output_path = 'C:\\Users\\Safiyyah\\eclipse-workspace\\tfidf.csv'
tfidf_df.to_csv(tfidf_output_path, index=True)
print(f"TF-IDF data successfully saved to {tfidf_output_path}")

# Bag-of-Words representation with adjusted max_df and min_df
bow_vectorizer = CountVectorizer(max_features=10000, max_df=0.85, min_df=2)
bow_matrix = bow_vectorizer.fit_transform(df['CLEANED_TEXT'])
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vectorizer.get_feature_names_out())

# Print the BoW Matrix
print("\nBoW Matrix:\n", bow_df)

# Save both representations to CSV files
bow_output_path = 'C:\\Users\\Safiyyah\\eclipse-workspace\\bow.csv'
bow_df.to_csv(bow_output_path, index=False)
print(f"Bag-of-Words data successfully saved to {bow_output_path}")
