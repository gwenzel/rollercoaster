"""
Review Data Processor for Captain Coaster
Cleans, processes, and prepares review data for ML training

Features:
- Remove URLs and unnecessary columns
- Sentiment analysis on review text
- Feature engineering for ML
- Export in ML-friendly formats
"""

import pandas as pd
import numpy as np
import re
from datetime import datetime
import json


class ReviewDataProcessor:
    """Process and clean Captain Coaster review data for ML training"""
    
    def __init__(self, csv_file_or_folder):
        """
        Initialize processor with CSV file or folder
        
        Args:
            csv_file_or_folder: Path to a single CSV file or folder containing CSV files
        """
        self.input_path = csv_file_or_folder
        self.df = None
        self.processed_df = None
        
    def load_data(self):
        """Load CSV data from file or combine multiple files from folder"""
        import glob
        import os
        
        # Check if input is a directory
        if os.path.isdir(self.input_path):
            print(f"Loading all CSV files from folder: {self.input_path}")
            
            # Find all CSV files in the folder
            csv_pattern = os.path.join(self.input_path, "*.csv")
            csv_files = glob.glob(csv_pattern)
            
            # Filter out processed files (to avoid processing already processed data)
            csv_files = [f for f in csv_files if 'processed_' not in os.path.basename(f)]
            
            if not csv_files:
                print(f"No CSV files found in {self.input_path}")
                return self
            
            print(f"Found {len(csv_files)} CSV files")
            
            # Load all files
            dfs = []
            for csv_file in csv_files:
                try:
                    df_temp = pd.read_csv(csv_file)
                    dfs.append(df_temp)
                    print(f"  ✓ Loaded {os.path.basename(csv_file)}: {len(df_temp)} reviews")
                except Exception as e:
                    print(f"  ✗ Error loading {os.path.basename(csv_file)}: {e}")
            
            if not dfs:
                print("No data loaded successfully")
                return self
            
            # Combine all dataframes
            print(f"\nCombining {len(dfs)} files...")
            self.df = pd.concat(dfs, ignore_index=True)
            
            # Remove duplicates based on review_id
            if 'review_id' in self.df.columns:
                before_dedup = len(self.df)
                self.df = self.df.drop_duplicates(subset=['review_id'], keep='first')
                duplicates_removed = before_dedup - len(self.df)
                if duplicates_removed > 0:
                    print(f"  Removed {duplicates_removed} duplicate reviews")
            
            print(f"✓ Combined dataset: {len(self.df)} unique reviews")
            
        else:
            # Single file
            print(f"Loading data from {self.input_path}...")
            self.df = pd.read_csv(self.input_path)
            print(f"✓ Loaded {len(self.df)} reviews")
        
        return self
    
    def clean_data(self):
        """Remove URLs and clean up data"""
        print("\nCleaning data...")
        
        # Create a copy for processing
        self.processed_df = self.df.copy()
        
        # Remove URL columns (keep IDs for reference)
        url_columns = [col for col in self.processed_df.columns if 'url' in col.lower()]
        print(f"  Removing URL columns: {', '.join(url_columns)}")
        self.processed_df = self.processed_df.drop(columns=url_columns, errors='ignore')
        
        # Remove scraped_at timestamp (not useful for ML)
        if 'scraped_at' in self.processed_df.columns:
            self.processed_df = self.processed_df.drop(columns=['scraped_at'])
        
        # Clean review text
        if 'review_text' in self.processed_df.columns:
            # Remove extra whitespace
            self.processed_df['review_text'] = self.processed_df['review_text'].str.strip()
            # Remove rows with no review text
            before_count = len(self.processed_df)
            self.processed_df = self.processed_df[self.processed_df['review_text'].notna()]
            self.processed_df = self.processed_df[self.processed_df['review_text'].str.len() > 0]
            removed = before_count - len(self.processed_df)
            if removed > 0:
                print(f"  Removed {removed} reviews with no text")
        
        # Fill NaN values in tags with empty string
        if 'tags_positive' in self.processed_df.columns:
            self.processed_df['tags_positive'] = self.processed_df['tags_positive'].fillna('')
        if 'tags_negative' in self.processed_df.columns:
            self.processed_df['tags_negative'] = self.processed_df['tags_negative'].fillna('')
        
        # Convert time_posted to a more structured format (extract relative time info)
        if 'time_posted' in self.processed_df.columns:
            self.processed_df['time_category'] = self.processed_df['time_posted'].apply(
                self._categorize_time
            )
        
        print(f"✓ Cleaned {len(self.processed_df)} reviews")
        return self
    
    def _categorize_time(self, time_str):
        """Categorize relative time into buckets"""
        if pd.isna(time_str):
            return 'unknown'
        time_str = str(time_str).lower()
        if 'minute' in time_str or 'second' in time_str:
            return 'very_recent'
        elif 'hour' in time_str:
            return 'recent'
        elif 'day' in time_str:
            return 'this_week'
        elif 'week' in time_str:
            return 'this_month'
        else:
            return 'older'
    
    def analyze_sentiment(self):
        """
        Perform sentiment analysis on review text
        Uses a simple rule-based approach (can be upgraded with transformers/VADER)
        """
        print("\nAnalyzing sentiment...")
        
        if 'review_text' not in self.processed_df.columns:
            print("  No review text found, skipping sentiment analysis")
            return self
        
        # Simple sentiment lexicon (can be expanded)
        positive_words = {
            'amazing', 'awesome', 'excellent', 'fantastic', 'great', 'love', 'best',
            'incredible', 'wonderful', 'perfect', 'epic', 'brilliant', 'outstanding',
            'superb', 'spectacular', 'thrilling', 'exhilarating', 'fun', 'smooth',
            'intense', 'legendary', 'phenomenal', 'beautiful', 'gorgeous', 'blast'
        }
        
        negative_words = {
            'bad', 'terrible', 'awful', 'worst', 'hate', 'boring', 'disappointing',
            'rough', 'painful', 'uncomfortable', 'slow', 'lame', 'meh', 'mediocre',
            'garbage', 'trash', 'poor', 'weak', 'dull', 'rattle', 'shake', 'jerk',
            'headache', 'hurt', 'violent', 'rough', 'bumpy'
        }
        
        def calculate_sentiment(text):
            if pd.isna(text) or not text:
                return 0.0
            
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)
            
            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            
            # Calculate sentiment score (-1 to 1)
            total = pos_count + neg_count
            if total == 0:
                return 0.0
            
            sentiment = (pos_count - neg_count) / max(len(words), 1)
            return np.clip(sentiment, -1, 1)
        
        self.processed_df['sentiment_score'] = self.processed_df['review_text'].apply(
            calculate_sentiment
        )
        
        # Categorize sentiment
        def categorize_sentiment(score):
            if score > 0.02:
                return 'positive'
            elif score < -0.02:
                return 'negative'
            else:
                return 'neutral'
        
        self.processed_df['sentiment_category'] = self.processed_df['sentiment_score'].apply(
            categorize_sentiment
        )
        
        # Statistics
        sentiment_dist = self.processed_df['sentiment_category'].value_counts()
        print(f"  Sentiment distribution:")
        for category, count in sentiment_dist.items():
            pct = (count / len(self.processed_df)) * 100
            print(f"    {category}: {count} ({pct:.1f}%)")
        
        print(f"  Average sentiment: {self.processed_df['sentiment_score'].mean():.3f}")
        
        return self
    
    def engineer_features(self):
        """Create ML-friendly features"""
        print("\nEngineering features...")
        
        # 1. Text-based features
        if 'review_text' in self.processed_df.columns:
            self.processed_df['review_length'] = self.processed_df['review_text'].str.len()
            self.processed_df['review_word_count'] = self.processed_df['review_text'].str.split().str.len()
            print(f"  ✓ Text features: length, word_count")
        
        # 2. Tag features - count positive and negative tags
        if 'tags_positive' in self.processed_df.columns:
            self.processed_df['num_positive_tags'] = self.processed_df['tags_positive'].apply(
                lambda x: len([t for t in str(x).split(',') if t.strip()]) if x else 0
            )
        
        if 'tags_negative' in self.processed_df.columns:
            self.processed_df['num_negative_tags'] = self.processed_df['tags_negative'].apply(
                lambda x: len([t for t in str(x).split(',') if t.strip()]) if x else 0
            )
        
        # Tag balance ratio
        if 'num_positive_tags' in self.processed_df.columns and 'num_negative_tags' in self.processed_df.columns:
            self.processed_df['tag_balance'] = (
                self.processed_df['num_positive_tags'] - self.processed_df['num_negative_tags']
            )
            print(f"  ✓ Tag features: counts and balance")
        
        # 3. Rating normalized to 0-1 scale
        if 'rating' in self.processed_df.columns:
            self.processed_df['rating_normalized'] = self.processed_df['rating'] / 5.0
            print(f"  ✓ Rating normalized (0-1)")
        
        # 4. Engagement score (combines rating and upvotes)
        if 'rating' in self.processed_df.columns and 'upvotes' in self.processed_df.columns:
            # Log transform upvotes to reduce skew
            self.processed_df['upvotes_log'] = np.log1p(self.processed_df['upvotes'])
            self.processed_df['engagement_score'] = (
                self.processed_df['rating_normalized'] * 0.7 + 
                (self.processed_df['upvotes_log'] / 5.0).clip(0, 1) * 0.3
            )
            print(f"  ✓ Engagement score")
        
        # 5. Has review flag (for filtering)
        self.processed_df['has_detailed_review'] = (
            self.processed_df['review_word_count'] > 5
        ) if 'review_word_count' in self.processed_df.columns else False
        
        print(f"✓ Engineered {len([c for c in self.processed_df.columns if c not in self.df.columns])} new features")
        
        return self
    
    def create_tag_features(self):
        """Create one-hot encoding for common tags"""
        print("\nCreating tag features...")
        
        # Collect all unique tags
        all_positive_tags = set()
        all_negative_tags = set()
        
        if 'tags_positive' in self.processed_df.columns:
            for tags in self.processed_df['tags_positive'].dropna():
                if tags:
                    all_positive_tags.update([t.strip().lower() for t in str(tags).split(',')])
        
        if 'tags_negative' in self.processed_df.columns:
            for tags in self.processed_df['tags_negative'].dropna():
                if tags:
                    all_negative_tags.update([t.strip().lower() for t in str(tags).split(',')])
        
        # Remove empty strings
        all_positive_tags.discard('')
        all_negative_tags.discard('')
        
        print(f"  Found {len(all_positive_tags)} unique positive tags")
        print(f"  Found {len(all_negative_tags)} unique negative tags")
        
        # Create binary features for common tags (appearing in at least 1% of reviews)
        min_count = max(1, int(len(self.processed_df) * 0.01))
        
        # Count tag frequencies
        tag_counts_pos = {}
        tag_counts_neg = {}
        
        for tags in self.processed_df['tags_positive'].dropna():
            if tags:
                for tag in str(tags).split(','):
                    tag = tag.strip().lower()
                    if tag:
                        tag_counts_pos[tag] = tag_counts_pos.get(tag, 0) + 1
        
        for tags in self.processed_df['tags_negative'].dropna():
            if tags:
                for tag in str(tags).split(','):
                    tag = tag.strip().lower()
                    if tag:
                        tag_counts_neg[tag] = tag_counts_neg.get(tag, 0) + 1
        
        # Filter common tags
        common_pos_tags = [tag for tag, count in tag_counts_pos.items() if count >= min_count]
        common_neg_tags = [tag for tag, count in tag_counts_neg.items() if count >= min_count]
        
        print(f"  Creating features for {len(common_pos_tags)} common positive tags")
        print(f"  Creating features for {len(common_neg_tags)} common negative tags")
        
        # Create binary columns
        for tag in common_pos_tags:
            col_name = f"tag_pos_{tag.replace(' ', '_').replace('-', '_')}"
            self.processed_df[col_name] = self.processed_df['tags_positive'].apply(
                lambda x: 1 if tag in str(x).lower() else 0
            )
        
        for tag in common_neg_tags:
            col_name = f"tag_neg_{tag.replace(' ', '_').replace('-', '_')}"
            self.processed_df[col_name] = self.processed_df['tags_negative'].apply(
                lambda x: 1 if tag in str(x).lower() else 0
            )
        
        return self
    
    def get_summary_statistics(self):
        """Generate summary statistics"""
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        print(f"\nDataset Size: {len(self.processed_df)} reviews")
        
        if 'rating' in self.processed_df.columns:
            print(f"\nRating Statistics:")
            print(f"  Mean: {self.processed_df['rating'].mean():.2f}")
            print(f"  Median: {self.processed_df['rating'].median():.2f}")
            print(f"  Std: {self.processed_df['rating'].std():.2f}")
            print(f"  Min: {self.processed_df['rating'].min():.2f}")
            print(f"  Max: {self.processed_df['rating'].max():.2f}")
        
        if 'upvotes' in self.processed_df.columns:
            print(f"\nUpvotes Statistics:")
            print(f"  Mean: {self.processed_df['upvotes'].mean():.2f}")
            print(f"  Median: {self.processed_df['upvotes'].median():.2f}")
            print(f"  Max: {self.processed_df['upvotes'].max()}")
        
        if 'sentiment_score' in self.processed_df.columns:
            print(f"\nSentiment Statistics:")
            print(f"  Mean: {self.processed_df['sentiment_score'].mean():.3f}")
            print(f"  Positive reviews: {(self.processed_df['sentiment_category'] == 'positive').sum()}")
            print(f"  Neutral reviews: {(self.processed_df['sentiment_category'] == 'neutral').sum()}")
            print(f"  Negative reviews: {(self.processed_df['sentiment_category'] == 'negative').sum()}")
        
        if 'coaster_name' in self.processed_df.columns:
            print(f"\nUnique Entities:")
            print(f"  Coasters: {self.processed_df['coaster_name'].nunique()}")
            if 'park_name' in self.processed_df.columns:
                print(f"  Parks: {self.processed_df['park_name'].nunique()}")
            if 'reviewer_name' in self.processed_df.columns:
                print(f"  Reviewers: {self.processed_df['reviewer_name'].nunique()}")
        
        print(f"\nTotal Features: {len(self.processed_df.columns)}")
        
        return self
    
    def save_processed_data(self, output_prefix="processed_reviews"):
        """Save processed data in multiple formats"""
        print(f"\nSaving processed data...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Save full processed CSV
        csv_file = f"{output_prefix}_{timestamp}.csv"
        self.processed_df.to_csv(csv_file, index=False)
        print(f"  ✓ Saved CSV: {csv_file}")
        
        # 2. Save ML-ready dataset (numeric features only)
        numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns.tolist()
        ml_df = self.processed_df[numeric_cols].copy()
        ml_file = f"{output_prefix}_ml_{timestamp}.csv"
        ml_df.to_csv(ml_file, index=False)
        print(f"  ✓ Saved ML-ready CSV: {ml_file} ({len(numeric_cols)} features)")
        
        # 3. Save metadata separately
        text_cols = ['coaster_name', 'park_name', 'reviewer_name', 'reviewer_id', 
                     'coaster_id', 'park_id', 'review_id', 'review_text',
                     'tags_positive', 'tags_negative', 'time_posted']
        metadata_cols = [c for c in text_cols if c in self.processed_df.columns]
        if metadata_cols:
            metadata_df = self.processed_df[metadata_cols].copy()
            metadata_file = f"{output_prefix}_metadata_{timestamp}.csv"
            metadata_df.to_csv(metadata_file, index=False)
            print(f"  ✓ Saved metadata: {metadata_file}")
        
        # 4. Save feature info as JSON
        feature_info = {
            'total_samples': len(self.processed_df),
            'total_features': len(self.processed_df.columns),
            'numeric_features': len(numeric_cols),
            'numeric_feature_names': numeric_cols,
            'metadata_features': metadata_cols,
            'rating_stats': {
                'mean': float(self.processed_df['rating'].mean()) if 'rating' in self.processed_df.columns else None,
                'std': float(self.processed_df['rating'].std()) if 'rating' in self.processed_df.columns else None,
            },
            'processing_date': timestamp
        }
        
        info_file = f"{output_prefix}_info_{timestamp}.json"
        with open(info_file, 'w') as f:
            json.dump(feature_info, f, indent=2)
        print(f"  ✓ Saved feature info: {info_file}")
        
        return {
            'csv': csv_file,
            'ml_csv': ml_file,
            'metadata': metadata_file if metadata_cols else None,
            'info': info_file
        }
    
    def process_all(self, output_prefix="processed_reviews"):
        """Run full processing pipeline"""
        print("\n" + "="*60)
        print("CAPTAIN COASTER REVIEW PROCESSOR")
        print("="*60)
        
        self.load_data()
        self.clean_data()
        self.analyze_sentiment()
        self.engineer_features()
        self.create_tag_features()
        self.get_summary_statistics()
        files = self.save_processed_data(output_prefix)
        
        print("\n" + "="*60)
        print("✓ PROCESSING COMPLETE!")
        print("="*60)
        print("\nGenerated files:")
        for key, path in files.items():
            if path:
                print(f"  {key}: {path}")
        
        return self.processed_df, files


def main():
    """Main execution"""
    import sys
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python process_reviews.py <input_csv_file_or_folder>")
        print("\nExamples:")
        print("  # Process a single file")
        print("  python process_reviews.py data/captaincoaster_small.csv")
        print("\n  # Process all CSV files in a folder")
        print("  python process_reviews.py data/")
        print("  python process_reviews.py data")
        print("\n  # Use current folder's data directory")
        print("  python process_reviews.py data")
        return
    
    input_path = sys.argv[1]
    
    # Check if path exists
    if not os.path.exists(input_path):
        print(f"Error: Path '{input_path}' not found!")
        print("\nTrying common locations...")
        
        # Try data subdirectory
        data_path = os.path.join('data', input_path)
        if os.path.exists(data_path):
            input_path = data_path
            print(f"Found: {input_path}")
        else:
            # Try just 'data' folder
            if os.path.exists('data'):
                input_path = 'data'
                print(f"Using default data folder: {input_path}")
            else:
                print(f"Path not found. Please check the path.")
                return
    
    # Determine output prefix and directory
    if os.path.isdir(input_path):
        # Processing a folder
        output_dir = input_path
        base_name = "all_reviews"
    else:
        # Processing a single file
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        output_dir = os.path.dirname(input_path) if os.path.dirname(input_path) else 'data'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_prefix = os.path.join(output_dir, f"processed_{base_name}")
    
    # Process the data
    processor = ReviewDataProcessor(input_path)
    processed_df, files = processor.process_all(output_prefix)
    
    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print(f"\n1. Use '{files['ml_csv']}' for ML training")
    print(f"2. Refer to '{files['info']}' for feature descriptions")
    print(f"3. Use '{files['metadata']}' to join back text/IDs if needed")
    print("\nReady for machine learning! 🚀")


if __name__ == "__main__":
    main()
