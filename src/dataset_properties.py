import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Set options to display all columns without truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# Load the dataset from the specified file path
file_path = 'C:\\Users\\Safiyyah\\eclipse-workspace\\ASSIGN-NEW ECOMMERCE WOMAN CLOTHING.xlsx'
df = pd.read_excel(file_path)

# 1) Dataset Structure
print("\nDataset Structure:")
df.info()

# 2) Summary Statistics for Classification Target Class
print("\nSummary Statistics for RECOMMENDED IND:")
if 'RECOMMENDED IND' in df.columns:
    print(df['RECOMMENDED IND'].describe())
else:
    print("Column 'RECOMMENDED IND' not found in the dataset.")

# 3) Distribution of Classification Target Class
plt.figure(figsize=(8, 5))
sns.countplot(x='RECOMMENDED IND', data=df, palette='viridis')
plt.title('Distribution of Recommended Indicator')
plt.xlabel('Recommended (1 = Yes, 0 = No)')
plt.ylabel('Count')
plt.show()

# 4) Text Data Properties
df['Review Length'] = df['REVIEW TEXT'].dropna().apply(lambda x: len(x.split()))
print("\nSummary Statistics for Review Length:")
print(df['Review Length'].describe())
all_words = df['REVIEW TEXT'].dropna().str.cat(sep=' ').split()
unique_words_count = len(set(all_words))
print(f"\nNumber of Unique Words in Review Text: {unique_words_count}")
print("\nUnique Values in Categorical Fields:")
print("Division Names:", df['DIVISION NAME'].nunique(), "->", df['DIVISION NAME'].unique())
print("Department Names:", df['DEPARTMENT NAME'].nunique(), "->", df['DEPARTMENT NAME'].unique())
print("Class Names:", df['CLASS NAME'].nunique(), "->", df['CLASS NAME'].unique())

# 5) Zipf's Law Distribution
# Calculate word frequencies and display the Zipf distribution
word_counts = Counter(all_words)
word_freq = pd.DataFrame(word_counts.items(), columns=['Word', 'Frequency']).sort_values(by='Frequency', ascending=False)
word_freq['Rank'] = range(1, len(word_freq) + 1)

# Define cutoff values for head and tail
head_cutoff = 100  # Words with rank <= 100
tail_cutoff = len(word_freq) * 0.95  # Top 5% rarest words

# Identify head and tail words
head_words = word_freq[word_freq['Rank'] <= head_cutoff]
tail_words = word_freq[word_freq['Rank'] >= tail_cutoff]

# Print top 10 head and tail words
print("\nTop 10 Head Words (frequent):")
print(head_words.head(10))
print("\nTop 10 Tail Words (rare):")
print(tail_words.tail(10))

# Plot Zipf's Law Distribution with colored head and tail words
plt.figure(figsize=(10, 6))
plt.plot(word_freq['Rank'], word_freq['Frequency'], color='purple', label="All Words", alpha=0.6)
plt.scatter(head_words['Rank'], head_words['Frequency'], color='red', label="Head Words (Frequent)")
plt.scatter(tail_words['Rank'], tail_words['Frequency'], color='blue', label="Tail Words (Rare)")

# Plot the head and tail cutoffs
plt.axvline(x=head_cutoff, color='red', linestyle='--', label="Head Cutoff")
plt.axvline(x=tail_cutoff, color='blue', linestyle='--', label="Tail Cutoff")

# Plot customization
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Rank of Word (log scale)')
plt.ylabel('Frequency of Word (log scale)')
plt.title("Zipf's Law Distribution of Word Frequencies with Head and Tail Highlighted")
plt.legend()
plt.show()



