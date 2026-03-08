Product Review Sentiment Analysis
A Python data science pipeline that classifies product reviews as positive or negative using TF-IDF vectorization and Logistic Regression. It handles the full workflow from text cleaning and model training to evaluation and visualization — outputting a six-panel dashboard covering ROC curves, confusion matrices, and top sentiment-driving words. Easily swappable to run on real Amazon, Yelp, or IMDB review datasets with minimal code changes.

Features

Full text preprocessing (lowercasing, HTML stripping, punctuation removal)
TF-IDF vectorization with unigram + bigram support
Logistic Regression classifier with scikit-learn Pipeline
Evaluation metrics: accuracy, ROC-AUC, precision, recall, F1
Six-panel visualization dashboard exported as PNG
Custom review inference with confidence scores
Plug-and-play support for real public datasets


Requirements
bashpip install scikit-learn numpy pandas matplotlib seaborn
To use a real dataset, also install:
bashpip install datasets

Usage
bashpython sentiment_analysis.py
The script runs end-to-end and outputs:

Console: training time, accuracy, ROC-AUC, classification report, and per-review predictions
sentiment_dashboard.png: six-panel visual summary of model performance
