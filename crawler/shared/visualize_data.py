"""
Quick data visualization for processed review data
Generates charts and statistics to understand the dataset
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

def visualize_processed_data(csv_file):
    """Create visualizations for processed review data"""
    
    print(f"Loading data from {csv_file}...")
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} reviews with {len(df.columns)} features\n")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Rating Distribution
    ax1 = plt.subplot(3, 3, 1)
    if 'rating' in df.columns:
        df['rating'].hist(bins=10, ax=ax1, color='skyblue', edgecolor='black')
        ax1.set_title('Rating Distribution', fontweight='bold')
        ax1.set_xlabel('Rating (stars)')
        ax1.set_ylabel('Count')
        ax1.axvline(df['rating'].mean(), color='red', linestyle='--', label=f'Mean: {df["rating"].mean():.2f}')
        ax1.legend()
    
    # 2. Sentiment Distribution
    ax2 = plt.subplot(3, 3, 2)
    if 'sentiment_category' in df.columns:
        sentiment_counts = df['sentiment_category'].value_counts()
        colors = {'positive': 'green', 'neutral': 'gray', 'negative': 'red'}
        bars = ax2.bar(sentiment_counts.index, sentiment_counts.values, 
                      color=[colors.get(x, 'blue') for x in sentiment_counts.index])
        ax2.set_title('Sentiment Distribution', fontweight='bold')
        ax2.set_xlabel('Sentiment')
        ax2.set_ylabel('Count')
        
        # Add percentages
        total = sentiment_counts.sum()
        for i, (idx, val) in enumerate(sentiment_counts.items()):
            pct = (val/total)*100
            ax2.text(i, val, f'{pct:.1f}%', ha='center', va='bottom')
    
    # 3. Sentiment Score vs Rating
    ax3 = plt.subplot(3, 3, 3)
    if 'sentiment_score' in df.columns and 'rating' in df.columns:
        ax3.scatter(df['sentiment_score'], df['rating'], alpha=0.5, s=10)
        ax3.set_title('Sentiment Score vs Rating', fontweight='bold')
        ax3.set_xlabel('Sentiment Score')
        ax3.set_ylabel('Rating (stars)')
        
        # Add correlation
        corr = df[['sentiment_score', 'rating']].corr().iloc[0, 1]
        ax3.text(0.05, 0.95, f'Correlation: {corr:.3f}', 
                transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Review Length Distribution
    ax4 = plt.subplot(3, 3, 4)
    if 'review_word_count' in df.columns:
        df['review_word_count'].hist(bins=30, ax=ax4, color='lightcoral', edgecolor='black')
        ax4.set_title('Review Length Distribution', fontweight='bold')
        ax4.set_xlabel('Word Count')
        ax4.set_ylabel('Count')
        ax4.set_xlim(0, df['review_word_count'].quantile(0.95))  # Trim outliers for vis
    
    # 5. Tag Balance Distribution
    ax5 = plt.subplot(3, 3, 5)
    if 'tag_balance' in df.columns:
        df['tag_balance'].hist(bins=20, ax=ax5, color='lightgreen', edgecolor='black')
        ax5.set_title('Tag Balance (Pos - Neg)', fontweight='bold')
        ax5.set_xlabel('Tag Balance')
        ax5.set_ylabel('Count')
        ax5.axvline(0, color='black', linestyle='--', linewidth=1)
    
    # 6. Upvotes Distribution
    ax6 = plt.subplot(3, 3, 6)
    if 'upvotes' in df.columns:
        upvote_counts = df['upvotes'].value_counts().sort_index().head(10)
        ax6.bar(upvote_counts.index.astype(str), upvote_counts.values, color='orange')
        ax6.set_title('Upvotes Distribution (Top 10)', fontweight='bold')
        ax6.set_xlabel('Number of Upvotes')
        ax6.set_ylabel('Count')
    
    # 7. Engagement Score Distribution
    ax7 = plt.subplot(3, 3, 7)
    if 'engagement_score' in df.columns:
        df['engagement_score'].hist(bins=30, ax=ax7, color='mediumpurple', edgecolor='black')
        ax7.set_title('Engagement Score Distribution', fontweight='bold')
        ax7.set_xlabel('Engagement Score')
        ax7.set_ylabel('Count')
    
    # 8. Top Coasters by Review Count
    ax8 = plt.subplot(3, 3, 8)
    if 'coaster_name' in df.columns:
        top_coasters = df['coaster_name'].value_counts().head(10)
        ax8.barh(range(len(top_coasters)), top_coasters.values, color='steelblue')
        ax8.set_yticks(range(len(top_coasters)))
        ax8.set_yticklabels(top_coasters.index, fontsize=8)
        ax8.set_title('Top 10 Most Reviewed Coasters', fontweight='bold')
        ax8.set_xlabel('Number of Reviews')
        ax8.invert_yaxis()
    
    # 9. Correlation Heatmap (numeric features)
    ax9 = plt.subplot(3, 3, 9)
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    # Select key features for correlation
    key_features = ['rating', 'sentiment_score', 'upvotes', 'review_word_count', 
                   'num_positive_tags', 'num_negative_tags', 'engagement_score']
    available_features = [f for f in key_features if f in numeric_cols]
    
    if len(available_features) > 1:
        corr_matrix = df[available_features].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, ax=ax9, square=True, cbar_kws={'shrink': 0.8})
        ax9.set_title('Feature Correlations', fontweight='bold')
        plt.setp(ax9.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.setp(ax9.get_yticklabels(), rotation=0, fontsize=8)
    
    plt.tight_layout()
    
    # Save figure
    output_file = csv_file.replace('.csv', '_visualization.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    if 'rating' in df.columns:
        print(f"\nRating Statistics:")
        print(f"  Count: {df['rating'].count()}")
        print(f"  Mean: {df['rating'].mean():.2f}")
        print(f"  Median: {df['rating'].median():.2f}")
        print(f"  Std: {df['rating'].std():.2f}")
        print(f"  Min: {df['rating'].min():.2f}")
        print(f"  Max: {df['rating'].max():.2f}")
    
    if 'sentiment_score' in df.columns:
        print(f"\nSentiment Statistics:")
        print(f"  Mean: {df['sentiment_score'].mean():.3f}")
        print(f"  Median: {df['sentiment_score'].median():.3f}")
        print(f"  Std: {df['sentiment_score'].std():.3f}")
    
    if 'sentiment_category' in df.columns:
        print(f"\nSentiment Distribution:")
        for cat, count in df['sentiment_category'].value_counts().items():
            pct = (count / len(df)) * 100
            print(f"  {cat}: {count} ({pct:.1f}%)")
    
    if 'review_word_count' in df.columns:
        print(f"\nReview Length Statistics:")
        print(f"  Mean: {df['review_word_count'].mean():.1f} words")
        print(f"  Median: {df['review_word_count'].median():.0f} words")
        print(f"  Max: {df['review_word_count'].max():.0f} words")
    
    if 'coaster_name' in df.columns:
        print(f"\nDataset Coverage:")
        print(f"  Unique Coasters: {df['coaster_name'].nunique()}")
        if 'park_name' in df.columns:
            print(f"  Unique Parks: {df['park_name'].nunique()}")
        if 'reviewer_name' in df.columns:
            print(f"  Unique Reviewers: {df['reviewer_name'].nunique()}")
    
    print(f"\nFeature Count: {len(df.columns)}")
    print(f"Sample Count: {len(df)}")
    
    # Show plot
    plt.show()

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_data.py <processed_csv_file>")
        print("\nExample:")
        print("  python visualize_data.py data/processed_captaincoaster_small_20251110_155805.csv")
        return
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found!")
        return
    
    visualize_processed_data(csv_file)

if __name__ == "__main__":
    main()
