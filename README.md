# NLP Air France Reviews Analysis

This repository contains code, data, and resources for analyzing customer reviews of Air France. The project covers various NLP tasks such as sentiment analysis, topic modeling, clustering, and more.

## Table of Contents

- [File Structure](#file-structure)
- [Virtual Machines (Azure ML Studio)](#virtual-machines-azure-ml-studio)
- [GitHub Repository](#github-repository)
- [Benefits of Using Cloud Resources](#benefits-of-using-cloud-resources)
- [Dataset Summary](#dataset-summary)
- [Notebooks Overview](#notebooks-overview)
  - [1_data_exploration.ipynb](#1_data_explorationipynb)
  - [2_data_preprocessing.ipynb](#2_data_preprocessingipynb)
  - [3_vectorization.ipynb](#3_vectorizationipynb)
  - [4_topic_modeling.ipynb](#4_topic_modelingipynb)
  - [5_sentiment_cluster_analysis.ipynb](#5_sentiment_cluster_analysisipynb)

## File Structure

The project directory is structured as follows:

- **data:**
  - **interim:** Intermediate data files, including various serialized models and vectorized files like `bert_vectorized.pkl`, `bow_vectorized.pkl`, and `sbert_vectorized.pkl`.
  - **processed:** Processed data files, including `cluster_analysis`, `lda_topic_distribution.csv`, and `sentiment_analysis_results.csv`.
  - **raw:** Raw data files, including `airfrance_tripadvisor_reviews.csv` containing customer reviews.
- **models:** Pre-trained machine learning models like `bert_topic_model` and `word2vec_model` along with clustering models for LDA, NMF, and HDBSCAN.
- **notebooks:** Jupyter notebooks for various stages of the project, such as `1_data_exploration.ipynb`, `2_data_preprocessing.ipynb`, `3_vectorization.ipynb`, etc.
- **environment.yml:** The environment configuration for reproducibility, listing dependencies used for the project.

## Virtual Machines (Azure ML Studio)

The project utilizes two virtual machines for different tasks:

### NLP-VM-with-GPU:
- **GPU used for topic model training.**
- Virtual machine with 4 cores, 28 GB RAM, and a 176 GB disk.
- **Estimated cost:** $0.66/hr when running.
- **Public IP address:** 13.95.120.196.

### NLP-CPU-VM:
- **CPU used for preprocessing tasks.**
- Virtual machine with 4 cores, 32 GB RAM, and a 150 GB disk.
- **Estimated cost:** $0.35/hr when running.
- **Public IP address:** 13.94.136.53.

## GitHub Repository

The project code and resources are hosted on GitHub at [https://github.com/oskar-wolf/nlp_air_france_reviews](https://github.com/oskar-wolf/nlp_air_france_reviews).

## Benefits of Using Cloud Resources as a Student

- **Cost-Effective:** Only pay for the resources used, making it affordable to scale up or down as needed.
- **Access to Powerful Hardware:** Cloud platforms provide powerful GPUs and CPUs that would be difficult to access otherwise.
- **Reproducibility:** Cloud environments ensure consistent code execution and results.
- **Managed Infrastructure:** Cloud platforms handle infrastructure management, allowing students to focus on analysis rather than setup.
- **Collaboration:** GitHub enables easy collaboration with peers and mentors for group projects.

## Dataset Summary

The **Air France Reviews Dataset** contains customer feedback on Air France services, obtained ethically from apify.com. The dataset includes the following columns:

- **Review Text:** Full text of the review.
- **Rating:** A numerical score from 1 to 5, representing customer satisfaction.
- **Review Date:** Date the review was posted.
- **Sentiment:** The sentiment of the review (positive, neutral, negative).
- **Review Source:** Platform from which the review was obtained.
- **Language:** Language of the review.

### Dataset Applications in NLP

- **Sentiment Analysis:** Determine the sentiment of customer reviews (positive/negative/neutral).
- **Topic Modeling:** Identify recurring themes in the reviews.
- **Emotion Detection:** Detect emotions like frustration or satisfaction in reviews.
- **Keyword Extraction:** Identify important terms related to service, comfort, punctuality, etc.

## Notebooks Overview

### 1_data_exploration.ipynb

This notebook focuses on the initial exploration of the dataset. Tasks performed include:
- **Loading the dataset** and checking its structure.
- **Handling missing values** and **duplicates**.
- **Text preprocessing** including lowercasing, removing punctuation, tokenization, and lemmatization.
- **Exploratory Data Analysis (EDA)** with visualizations like word clouds and distributions of ratings.
- **Feature Engineering** and preparing the data for further analysis.

### 2_data_preprocessing.ipynb

The notebook handles data cleaning and preprocessing tasks:
- **Text Cleaning** with stopwords removal, punctuation, and number removal.
- **Tokenization and Lemmatization** using NLTK and SpaCy.
- **Feature Engineering** by adding date-related features like day, month, and year.
- **Data Mapping** to standardize expressions and improve analysis.

### 3_vectorization.ipynb

This notebook focuses on text vectorization using multiple models:
- **TF-IDF:** Converts text into numerical vectors based on term frequency.
- **Word2Vec:** Learns word associations from a large corpus of text.
- **BERT:** Uses a transformer-based model for contextual word embeddings.
- **Embedding Extraction:** For each method, embeddings are extracted and saved for further use.
- **Dimensionality Reduction (Optional):** PCA and t-SNE for visualizing embeddings.

### 4_topic_modeling.ipynb

This notebook applies several topic modeling techniques:
- **LDA (Latent Dirichlet Allocation):** A generative model for uncovering topics in a set of documents.
- **NMF (Non-negative Matrix Factorization):** Factorizes the document-term matrix to uncover topics.
- **BERTopic:** Leverages BERT embeddings for more context-sensitive topic modeling.
- **Word2Vec + KMeans:** Uses Word2Vec embeddings for semantic clustering with KMeans.
- **HDBSCAN:** A density-based clustering algorithm used with SBERT embeddings.

### 5_sentiment_cluster_analysis.ipynb

This notebook combines sentiment analysis with clustering:
- **Sentiment Analysis:** Uses VADER and BERT for sentiment prediction.
- **Clustering Analysis:** Applies clustering models like LDA, NMF, Word2Vec, HDBSCAN, and BERTopic to the dataset.
- **Model Evaluation:** Computes clustering metrics like Silhouette Score, Davies-Bouldin Index, and Intra-cluster Distance.
- **Visualization:** Visualizes clusters using UMAP and t-SNE.
- **Word Clouds:** Generates word clouds for each cluster to visualize the most frequent terms.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Requirements

To replicate the environment, install the dependencies listed in `environment.yml`.

```bash
conda env create -f environment.yml