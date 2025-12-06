🎬 Netflix Movies & TV Shows Clustering

Unsupervised Machine Learning Project

This project analyzes the Netflix Movies and TV Shows dataset and performs clustering to group similar content based on features such as genre, country, cast, duration, and description embeddings.

The goal is to explore hidden patterns within Netflix content and understand how different titles relate to each other using K-Means, Hierarchical Clustering, and DBSCAN.


🚀 Project Overview

In this project, we:

Clean and preprocess raw Netflix data

Perform feature engineering (genre vectorization, duration encoding, text processing)

Apply and compare three clustering algorithms

Reduce dimensionality using PCA and t-SNE

Visualize high-dimensional patterns

Interpret cluster meaning and content behavior

This is a perfect project for your Data Science portfolio, showcasing unsupervised ML skills.

Project Structure:
📁 netflix-clustering-project
│
├── data/
│   └── netflix_titles.csv
│
├── notebooks/
│   └── netflix_clustering.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── clustering_models.py
│   └── visualization.py
│
├── results/
│   ├── pca_plot.png
│   ├── tsne_plot.png
│   └── cluster_summary.csv
│
└── README.md


🧰 Tech Stack Used:
Languages

Python

Libraries

Pandas

NumPy

Scikit-learn

Matplotlib

Seaborn

NLTK / spaCy (optional)

SciPy


🧹 Data Preprocessing-

Key steps performed:

Removed duplicates

Handled missing values

Cleaned text fields (cast, director, description)

Converted date fields into Year, Month

Standardized duration (converted minutes → numeric)

Processed content type (Movie / TV Show)



🧪 Feature Engineering-

We created machine-friendly features:

✔ Genre Vectorization

Multi-label binarization of genre categories

Creates a high-dimensional genre matrix

✔ Country Encoding

Extract primary country

One-hot encoding

✔ Description Embeddings

TF-IDF vectorization

Reduces text into numerical vectors

✔ Duration Conversion

Movies → minutes

TV Shows → number of seasons



🤖 Clustering Models Used-
1️⃣ K-Means Clustering

Elbow method for choosing optimal K

Evaluated using inertia & silhouette score

Visualized using PCA scatter plots

2️⃣ Hierarchical Clustering

Agglomerative clustering

Dendrogram for visualizing cluster merging

Useful for understanding natural grouping

3️⃣ DBSCAN

Density-based clustering

Identifies noise/outliers

Good for uneven cluster shapes



📉 Dimensionality Reduction:
🔹 PCA (Principal Component Analysis)

Reduced high-dimensional vector space

Visualized top principal components

🔹 t-SNE

Captures non-linear relationships

Shows tight clusters based on description + genre

📊 Visualizations Included:

PCA cluster scatter plot

t-SNE cluster analysis

Genre distribution heatmap

Country-based clustering comparison

Dendrogram for hierarchical clustering

Elbow method graph for K-Means

Silhouette score visualization



📈 Insights & Interpretation-

Some possible outcomes:

Certain genres form tight clusters (e.g., Horror, Romance).

Indian, US, and UK content appear in separate clusters.

TV Shows and Movies cluster differently due to duration + structure.

DBSCAN detects outliers like niche documentaries.

K-Means gives stable clusters for content recommendation use-cases.



▶️ How to Run the Project-
1️⃣ Clone Repo
git clone https://github.com/yourusername/netflix-clustering.git

2️⃣ Install Requirements
pip install -r requirements.txt

3️⃣ Run Notebook

Open:

notebooks/netflix_clustering.ipynb

4️⃣ Run Standalone Scripts
python src/data_preprocessing.py
python src/feature_engineering.py
python src/clustering_models.py
python src/visualization.py

🧠 Use Cases of This Project

Content recommendation engine

Similar movie grouping

Catalog organization

Market segmentation

Language/genre-based insights



⭐ Future Enhancements:

Include deep learning embeddings (BERT, SentenceTransformers)

Deploy dashboard using Streamlit

Add similarity search (cosine similarity)

Build a Netflix-like recommendation system
