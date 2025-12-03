#Topic Modeling 
from sklearn.decomposition import LatentDirichletAllocation 
import pandas as pd 
import numpy as np 
 
# Load the saved Bag-of-Words matrix CSV file 
bow_file_path = "C:\\Users\\Safiyyah\\eclipse-workspace\\bow.csv" 
bow_df = pd.read_csv(bow_file_path) 
 
# Convert the BoW DataFrame to a numpy array for LDA 
bow_matrix = bow_df.values 
 
# Initialize and fit the LDA model 
# Increase the number of topics to make each topic more specific 
lda = LatentDirichletAllocation(n_components=8, random_state=42) 
lda.fit(bow_matrix) 
 
# Display topics and top words for each topic 
num_words = 10  # Number of words to show per topic 
 
# You will need to load the original CountVectorizer to get feature names 
# Load the vocabulary (feature names) from BoW vectorizer if available, otherwise list the top words 
feature_names = bow_df.columns  # This uses the column names from the BoW CSV file 
 
# Display topics 
for idx, topic in enumerate(lda.components_): 
    print(f"\nTopic {idx + 1}:") 
    print(" ".join([feature_names[i] for i in topic.argsort()[-num_words:]])) 
 
 
# Calculate topic coherence scores based on the BoW matrix 
top_n_words = 10  # Number of top words per topic to consider for coherence 
 
# Total number of documents in the dataset 
num_documents = bow_matrix.shape[0] 
topics = [] 
for topic_idx, topic in enumerate(lda.components_): 
    top_words = topic.argsort()[-top_n_words:][::-1] 
    topics.append(top_words) 
 
coherence_scores = [] 
for topic_idx, top_words in enumerate(topics): 
    score = 0.0 
    pairs = 0 
 
    # Compute pairwise word coherence 
    for i in range(len(top_words)): 
        for j in range(i + 1, len(top_words)): 
            word_i = bow_matrix[:, top_words[i]] 
            word_j = bow_matrix[:, top_words[j]] 
            co_occurrence = np.dot(word_i, word_j.T) / num_documents  # Normalize by document count 
            score += co_occurrence 
            pairs += 1 
 
    # Calculate coherence as an average score without scaling 
    coherence = (score / pairs) if pairs != 0 else 0 
    coherence_scores.append(coherence) 
    print(f"\nCoherence score for Topic {topic_idx + 1}: {coherence:.4f}") 
 
# Average coherence score across all topics 
average_coherence_score = np.mean(coherence_scores) 
print(f"\nAverage coherence score across all topics: {average_coherence_score:.4f}")