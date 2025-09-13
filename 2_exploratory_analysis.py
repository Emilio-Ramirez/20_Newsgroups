"""
Exploratory analysis module for 20 Newsgroups classification
Provides visualizations and statistical analysis of the dataset
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer

class NewsGroupsEDA:
    """
    Exploratory Data Analysis class for 20 Newsgroups dataset
    """
    
    def __init__(self, texts, labels, categories):
        """
        Initialize the EDA class
        
        Args:
            texts (list): List of text documents
            labels (list): List of labels/categories  
            categories (list): List of category names
        """
        self.texts = texts
        self.labels = labels
        self.categories = categories
        self.df = pd.DataFrame({
            'text': texts,
            'category': labels,
            'text_length': [len(text.split()) for text in texts],
            'char_length': [len(text) for text in texts]
        })
    
    def get_dataset_overview(self):
        """
        Get basic statistics about the dataset
        
        Returns:
            dict: Dataset overview statistics
        """
        overview = {
            'total_documents': len(self.texts),
            'total_categories': len(self.categories),
            'avg_text_length_words': self.df['text_length'].mean(),
            'median_text_length_words': self.df['text_length'].median(),
            'avg_char_length': self.df['char_length'].mean(),
            'median_char_length': self.df['char_length'].median(),
            'min_text_length': self.df['text_length'].min(),
            'max_text_length': self.df['text_length'].max(),
            'std_text_length': self.df['text_length'].std()
        }
        
        return overview
    
    def plot_category_distribution(self, title="Category Distribution in 20 Newsgroups Dataset"):
        """
        Plot the distribution of documents across categories
        
        Args:
            title (str): Title for the plot
            
        Returns:
            plotly.graph_objects.Figure: The distribution plot
        """
        category_counts = self.df['category'].value_counts()
        
        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            title=title,
            labels={'x': 'Category', 'y': 'Number of Documents'},
            color=category_counts.values,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Number of Documents",
            xaxis_tickangle=-45,
            height=600,
            showlegend=False
        )
        
        return fig
    
    def plot_text_length_distribution(self, title="Text Length Distribution"):
        """
        Plot histogram and boxplot of text lengths
        
        Args:
            title (str): Title for the plot
            
        Returns:
            plotly.graph_objects.Figure: The text length distribution plot
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Histogram of Text Lengths (Words)", "Boxplot by Category"],
            vertical_spacing=0.1
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=self.df['text_length'],
                nbinsx=50,
                name="Text Length Distribution",
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Boxplot by category
        for category in self.categories:
            category_lengths = self.df[self.df['category'] == category]['text_length']
            fig.add_trace(
                go.Box(
                    y=category_lengths,
                    name=category,
                    boxpoints='outliers'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Text Length (Words)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Category", row=2, col=1)
        fig.update_yaxes(title_text="Text Length (Words)", row=2, col=1)
        
        return fig
    
    def get_text_length_stats_by_category(self):
        """
        Get text length statistics grouped by category
        
        Returns:
            pandas.DataFrame: Statistics by category
        """
        stats = self.df.groupby('category')['text_length'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        
        stats = stats.sort_values('mean', ascending=False)
        return stats
    
    def analyze_word_frequency(self, top_n=20, min_word_length=3):
        """
        Analyze word frequency across the entire dataset
        
        Args:
            top_n (int): Number of top words to return
            min_word_length (int): Minimum word length to consider
            
        Returns:
            tuple: (word_counts, top_words_df)
        """
        # Combine all texts
        all_text = ' '.join(self.texts).lower()
        
        # Remove special characters and split into words
        words = re.findall(r'\b[a-z]+\b', all_text)
        
        # Filter by length and remove common stopwords
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'is', 'was', 'are', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'must', 'shall', 'can', 'this', 'that',
                     'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
                     'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her',
                     'its', 'our', 'their', 'from', 'up', 'out', 'if', 'about', 'into',
                     'through', 'during', 'before', 'after', 'above', 'below', 'between'}
        
        filtered_words = [word for word in words if len(word) >= min_word_length and word not in stopwords]
        
        # Count words
        word_counts = Counter(filtered_words)
        top_words = word_counts.most_common(top_n)
        
        # Create DataFrame
        top_words_df = pd.DataFrame(top_words, columns=['word', 'frequency'])
        
        return word_counts, top_words_df
    
    def plot_word_frequency(self, top_n=20, title="Most Frequent Words"):
        """
        Plot the most frequent words
        
        Args:
            top_n (int): Number of top words to plot
            title (str): Title for the plot
            
        Returns:
            plotly.graph_objects.Figure: Word frequency plot
        """
        word_counts, top_words_df = self.analyze_word_frequency(top_n)
        
        fig = px.bar(
            top_words_df,
            x='frequency',
            y='word',
            orientation='h',
            title=title,
            labels={'frequency': 'Frequency', 'word': 'Word'},
            color='frequency',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=600,
            showlegend=False
        )
        
        return fig
    
    def analyze_category_specific_words(self, category, top_n=15):
        """
        Analyze words specific to a particular category
        
        Args:
            category (str): Category to analyze
            top_n (int): Number of top words to return
            
        Returns:
            pandas.DataFrame: Top words for the category
        """
        category_texts = self.df[self.df['category'] == category]['text'].tolist()
        category_text = ' '.join(category_texts).lower()
        
        # Extract words
        words = re.findall(r'\b[a-z]+\b', category_text)
        
        # Filter words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        filtered_words = [word for word in words if len(word) >= 3 and word not in stopwords]
        
        # Count words
        word_counts = Counter(filtered_words)
        top_words = word_counts.most_common(top_n)
        
        return pd.DataFrame(top_words, columns=['word', 'frequency'])
    
    def plot_class_balance(self, title="Class Balance Analysis"):
        """
        Create a comprehensive class balance visualization
        
        Args:
            title (str): Title for the plot
            
        Returns:
            plotly.graph_objects.Figure: Class balance plot
        """
        category_counts = self.df['category'].value_counts()
        
        # Create subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Document Count by Category", "Class Distribution (Pie Chart)"],
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Bar plot
        fig.add_trace(
            go.Bar(
                x=category_counts.index,
                y=category_counts.values,
                name="Document Count",
                marker_color='lightcoral'
            ),
            row=1, col=1
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=category_counts.index,
                values=category_counts.values,
                name="Distribution"
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=title,
            height=500,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Category", tickangle=-45, row=1, col=1)
        fig.update_yaxes(title_text="Number of Documents", row=1, col=1)
        
        return fig
    
    def get_category_word_diversity(self):
        """
        Calculate vocabulary diversity for each category
        
        Returns:
            pandas.DataFrame: Vocabulary diversity statistics
        """
        diversity_stats = []
        
        for category in self.categories:
            category_texts = self.df[self.df['category'] == category]['text'].tolist()
            category_text = ' '.join(category_texts).lower()
            
            # Extract words
            words = re.findall(r'\b[a-z]+\b', category_text)
            unique_words = set(words)
            
            diversity_stats.append({
                'category': category,
                'total_words': len(words),
                'unique_words': len(unique_words),
                'vocabulary_diversity': len(unique_words) / len(words) if words else 0,
                'avg_word_frequency': len(words) / len(unique_words) if unique_words else 0
            })
        
        return pd.DataFrame(diversity_stats).sort_values('vocabulary_diversity', ascending=False)
    
    def create_comprehensive_overview(self):
        """
        Create a comprehensive overview of the dataset
        
        Returns:
            dict: Dictionary containing all analysis results
        """
        results = {
            'overview': self.get_dataset_overview(),
            'category_distribution': self.plot_category_distribution(),
            'text_length_distribution': self.plot_text_length_distribution(),
            'text_length_stats': self.get_text_length_stats_by_category(),
            'word_frequency': self.plot_word_frequency(),
            'class_balance': self.plot_class_balance(),
            'vocabulary_diversity': self.get_category_word_diversity()
        }
        
        return results
    
    def print_dataset_summary(self):
        """
        Print a comprehensive summary of the dataset
        """
        overview = self.get_dataset_overview()
        
        print("=" * 60)
        print("20 NEWSGROUPS DATASET SUMMARY")
        print("=" * 60)
        print(f"Total Documents: {overview['total_documents']:,}")
        print(f"Total Categories: {overview['total_categories']}")
        print(f"Average Text Length: {overview['avg_text_length_words']:.1f} words")
        print(f"Median Text Length: {overview['median_text_length_words']:.1f} words")
        print(f"Text Length Range: {overview['min_text_length']} - {overview['max_text_length']} words")
        print(f"Standard Deviation: {overview['std_text_length']:.1f} words")
        
        print("\nCATEGORY DISTRIBUTION:")
        print("-" * 40)
        category_counts = self.df['category'].value_counts()
        for category, count in category_counts.items():
            percentage = (count / len(self.texts)) * 100
            print(f"{category:<30} {count:>4} ({percentage:>5.1f}%)")
        
        print("\nTEXT LENGTH STATISTICS BY CATEGORY:")
        print("-" * 60)
        stats = self.get_text_length_stats_by_category()
        print(stats.to_string())
        
        print("\nTOP 10 MOST FREQUENT WORDS:")
        print("-" * 30)
        _, top_words_df = self.analyze_word_frequency(10)
        for _, row in top_words_df.iterrows():
            print(f"{row['word']:<15} {row['frequency']:>6}")
        
        print("\nVOCABULARY DIVERSITY BY CATEGORY:")
        print("-" * 50)
        diversity = self.get_category_word_diversity()
        for _, row in diversity.iterrows():
            print(f"{row['category']:<30} {row['vocabulary_diversity']:.4f}")

def main():
    """
    Main function to demonstrate the exploratory analysis
    """
    # This would typically be called from another script
    # For demonstration, we'll show how to use it
    print("Exploratory Analysis module loaded successfully!")
    print("To use this module:")
    print("1. Import: from exploratory_analysis import NewsGroupsEDA")
    print("2. Initialize: eda = NewsGroupsEDA(texts, labels, categories)")
    print("3. Run analysis: results = eda.create_comprehensive_overview()")
    print("4. Print summary: eda.print_dataset_summary()")

if __name__ == "__main__":
    main()