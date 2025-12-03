# Main file for running classification, clustering, and topic modeling
import subprocess

# Function to call the numerical script (TF-IDF and BoW processing)
def run_numerical_processing():
    print("\n======= Running Numerical Processing (TF-IDF & BoW) =======")
    subprocess.run(['python', 'numerical.py'], check=True)

# Function to call the classification script
def run_classification():
    print("\n======= Running Classification =======")
    subprocess.run(['python', 'classification.py'], check=True)

# Function to call the clustering script
def run_clustering():
    print("\n======= Running Clustering =======")
    subprocess.run(['python', 'clustering.py'], check=True)

# Function to call the topic modeling script
def run_topic_modeling():
    print("\n======= Running Topic Modeling =======")
    subprocess.run(['python', 'topic_modeling.py'], check=True)

# Main function to run all tasks
def main():
    print("\n======= Starting Text Mining Tasks =======")
    
    # Run numerical processing
    run_numerical_processing()
    
    # Run classification
    run_classification()
    
    # Run clustering
    run_clustering()
    
    # Run topic modeling
    run_topic_modeling()
    
    print("\n======= All tasks completed successfully =======")

# Entry point of the script
if __name__ == "__main__":
    main()
