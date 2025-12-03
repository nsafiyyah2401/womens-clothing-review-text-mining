import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.metrics.cluster import davies_bouldin_score 
from sklearn import metrics 
from sklearn.preprocessing import LabelEncoder 
from sklearn.decomposition import PCA 
 
# Load the preprocessed data and TF-IDF matrix 
df = pd.read_excel('C:\\Users\\Safiyyah\\eclipse-workspace\\processed_ASSIGN-ECOMMERCE WOMAN CLOTHING.xlsx')  
tfidf_matrix = pd.read_csv('C:\\Users\\Safiyyah\\eclipse-workspace\\TFIDF.csv', index_col=0)
tfidf_matrix.dropna(how='all', inplace=True)  # Drop rows that are completely empty 
 
# Encode 'Sentiment' labels for external evaluation 
label_encoder = LabelEncoder() 
true_labels = label_encoder.fit_transform(df['CLASS NAME']) 
 
# Define purity score 
def purity_score(y_true, y_pred): 
    contingency_matrix = metrics.cluster.pair_confusion_matrix(y_true, y_pred) 
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
 
# Visualization function for clustering (using PCA) 
def visualize_clusters_pca(tfidf_matrix, results, title="K-Means Clustering using PCA"): 
    """ 
    Visualize clusters in a 2D space using PCA for all K values. 
    """ 
    plt.figure(figsize=(12, 8)) 
 
    # Apply PCA on the TF-IDF matrix and plot for each K value 
    for k, y_kmeans in results: 
        # Apply PCA to reduce dimensionality 
        pca = PCA(n_components=2) 
        reduced_data = pca.fit_transform(tfidf_matrix) 
 
        # Plot the clusters 
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=y_kmeans, cmap='viridis', label=f"K={k}", alpha=0.7) 
 
    # Add titles and labels 
    plt.title(title) 
    plt.xlabel('PCA Component 1') 
    plt.ylabel('PCA Component 2') 
    plt.legend(loc='best') 
    plt.grid(True) 
    plt.show() 
 
# Set up variables to store results 
clustering_results = [] 
 
# Get feature names from the tfidf_matrix 
feature_names = tfidf_matrix.columns 
 
# Loop over a range of K values 
for k in range(2, 10):  # Adjust the range as needed 
    # Initialize KMeans with k clusters 
    km = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=42) 
    y_kmeans = km.fit_predict(tfidf_matrix) 
     
    # Collect the clustering results for visualization 
    clustering_results.append((k, y_kmeans)) 
 
    # Print top words for each cluster 
    print(f'\nTop words in each cluster for K={k}:') 
    for i, centroid in enumerate(km.cluster_centers_): 
        # Get indices of top N words for each cluster 
        top_indices = centroid.argsort()[-10:][::-1]  # Get indices of top 10 words 
        top_words = [feature_names[j] for j in top_indices] 
        print(f'Cluster {i}: {", ".join(top_words)}') 
 
# Call the PCA visualization function for all K 
visualize_clusters_pca(tfidf_matrix.to_numpy(), clustering_results)
