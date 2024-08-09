import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load the CSV file
file_path = 'outputs/merged_output.csv'  # Update with your file path
data = pd.read_csv(file_path)


# Preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = text.translate(str.maketrans('', '', string.punctuation + '0123456789'))
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)


# Apply preprocessing to the Abstract column
data['Processed Abstract'] = data['Abstract'].fillna('').apply(preprocess_text)

# Vectorize the text using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Processed Abstract'])

# Reduce dimensionality using PCA
pca = PCA(n_components=2)  # Set to 2 for visualization
pca_matrix = pca.fit_transform(tfidf_matrix.toarray())

# Perform clustering
num_clusters = 20  # Number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(pca_matrix)

# Assign cluster labels to the data
data['Cluster Label'] = kmeans.labels_

# Find the most central paper in each cluster
closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, pca_matrix)

# Collect key papers based on cluster centrality
key_papers = data.iloc[closest]

# If less than 200, add more based on distance to centroid
remaining_papers = 200 - len(key_papers)
if remaining_papers > 0:
    all_distances = kmeans.transform(pca_matrix)
    all_indices = np.argsort(all_distances, axis=0)
    additional_papers = []
    for i in range(num_clusters):
        cluster_indices = all_indices[:remaining_papers, i]
        additional_papers.extend(data.iloc[cluster_indices].index)
    additional_papers = list(set(additional_papers) - set(key_papers.index))[:remaining_papers]
    key_papers = pd.concat([key_papers, data.loc[additional_papers]])

# Save the top 200 papers to a CSV file
top_200_papers = key_papers.head(200)
top_200_papers.to_csv('outputs/top_200_papers.csv', index=False)

# Save the full dataset with cluster labels to a CSV file
data.to_csv('outputs/clustered_papers.csv', index=False)

# Plot the PCA results with clustering
plt.figure(figsize=(12, 8))
sns.scatterplot(x=pca_matrix[:, 0], y=pca_matrix[:, 1], hue=data['Cluster Label'], palette='viridis', s=50, legend=None)
plt.title('PCA of Paper Abstracts with Clustering')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.grid(True)
plt.savefig('outputs/clustering_visualization.png')  # Save the plot as an image file
plt.show()

# Print confirmation
print("Top 200 papers saved to 'top_200_papers.csv'")
print("Full clustering results saved to 'clustered_papers.csv'")
print("Clustering visualization saved to 'clustering_visualization.png'")
