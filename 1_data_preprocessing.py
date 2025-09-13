#Import libraries for text processing
import os
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

class NewsGroupsPreprocessor:
    
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.label_encoder = LabelEncoder()
        self.tfidf_vectorizer = None
        self.stop_words = None
        
        # Download nltk requirements first
        self.download_nltk_requirements()
        
        # Initialize stopwords after downloding
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            print("Using default stopwords")
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
    def download_nltk_requirements(self):
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            nltk.download('stopwords', quiet=True)
            print("NLTK requirements downloaded successfully")
        except Exception as e:
            print("Error downloading NLTK data")
    
    def load_dataset(self, data_path=None):
        if data_path:
            self.data_path = data_path
            
        if not self.data_path:
            raise ValueError("Data path must be provided")
            
        texts = []
        labels = []
        categories = []
        
        # Get all categorys directorys
        for category in sorted(os.listdir(self.data_path)):
            category_path = os.path.join(self.data_path, category)
            
            # Skip if not a directory
            if not os.path.isdir(category_path):
                continue
                
            categories.append(category)
            
            # Read all files in this category
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        texts.append(content)
                        labels.append(category)
                except Exception as e:
                    print("Error reading file")
                    continue
        
        print("Loaded " + str(len(texts)) + " documents from " + str(len(categories)) + " categories")
        print("Categories: " + str(categories))
        
        return texts, labels, categories
    
    def extract_content_from_newsgroup(self, text):
        # Split text into lines
        lines = text.split('\n')
        
        # Find where headers end
        content_start = 0
        for i, line in enumerate(lines):
            if line.strip() == '':
                content_start = i + 1
                break
        
        # Join the content lines
        content = '\n'.join(lines[content_start:])
        
        return content
    
    def clean_text(self, text):
        # Extract content remove headers
        text = self.extract_content_from_newsgroup(text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove email adresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove quoted text lines starting with >
        lines = text.split('\n')
        lines = [line for line in lines if not line.strip().startswith('>')]
        text = '\n'.join(lines)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        return text.strip()
    
    def tokenize_and_remove_stopwords(self, text):
        # Simple tokenization by spliting on whitespace
        tokens = text.split()
        
        # Remove stopwords and short words
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def preprocess_texts(self, texts, labels=None):
        print("Cleaning texts...")
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        print("Removing stopwords...")
        processed_texts = [self.tokenize_and_remove_stopwords(text) for text in cleaned_texts]
        
        # Keep track of non empty texts and their indices
        valid_indices = []
        valid_texts = []
        
        for i, text in enumerate(processed_texts):
            if text.strip():
                valid_indices.append(i)
                valid_texts.append(text)
        
        print("Kept " + str(len(valid_texts)) + " out of " + str(len(texts)) + " texts after preprocessing")
        
        if labels is not None:
            valid_labels = [labels[i] for i in valid_indices]
            return valid_texts, valid_labels
        
        return valid_texts
    
    def encode_labels(self, labels):
        encoded_labels = self.label_encoder.fit_transform(labels)
        print("Label encoding: " + str(dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))))
        return encoded_labels
    
    def vectorize_texts(self, texts, max_features=10000, test_size=0.2, random_state=42):
        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        
        print("Vectorizing texts with max_features=" + str(max_features) + "...")
        X = self.tfidf_vectorizer.fit_transform(texts)
        
        print("TF-IDF matrix shape: " + str(X.shape))
        print("Vocabulary size: " + str(len(self.tfidf_vectorizer.vocabulary_)))
        
        return X.toarray()
    
    def create_train_test_split(self, X, y, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print("Training set size: " + str(X_train.shape[0]) + " samples")
        print("Test set size: " + str(X_test.shape[0]) + " samples")
        print("Feature dimensions: " + str(X_train.shape[1]))
        
        return X_train, X_test, y_train, y_test
    
    def get_data_info(self, texts, labels):
        info = {
            'total_documents': len(texts),
            'categories': len(set(labels)),
            'category_names': sorted(list(set(labels))),
            'avg_text_length': np.mean([len(text.split()) for text in texts]),
            'min_text_length': min([len(text.split()) for text in texts]),
            'max_text_length': max([len(text.split()) for text in texts])
        }
        
        # Category distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        info['category_distribution'] = dict(zip(unique_labels, counts))
        
        return info

def main():
    # Initialize preprocessor
    data_path = "20_newsgroups"
    preprocessor = NewsGroupsPreprocessor(data_path)
    
    # Download NLTK requirements
    preprocessor.download_nltk_requirements()
    
    # Load dataset
    print("Loading dataset...")
    texts, labels, categories = preprocessor.load_dataset()
    
    # Get dataset info
    info = preprocessor.get_data_info(texts, labels)
    print("Dataset Information:")
    for key, value in info.items():
        if key != 'category_distribution':
            print(str(key) + ": " + str(value))
    
    print("Category Distribution:")
    for category, count in info['category_distribution'].items():
        print(str(category) + ": " + str(count))
    
    # Preprocess texts
    print("Preprocessing texts...")
    processed_texts, filtered_labels = preprocessor.preprocess_texts(texts, labels)
    
    # Encode labels
    encoded_labels = preprocessor.encode_labels(filtered_labels)
    
    # Vectorize texts
    X = preprocessor.vectorize_texts(processed_texts)
    
    # Create train test split
    X_train, X_test, y_train, y_test = preprocessor.create_train_test_split(X, encoded_labels)
    
    print("Preprocessing completed successfully!")
    print("Ready for neural network training with " + str(X_train.shape[1]) + " features")
    
    return X_train, X_test, y_train, y_test, preprocessor

if __name__ == "__main__":
    main()