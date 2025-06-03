# Project 1: Analysis of Nationalist Vocabulary in Olympic Coverage
# ================================================================
# This script analyzes nationalist language in Olympic coverage from 
# The New York Times (NYT) and People's Daily (China).
# It compares differences in nationalist expression between US and Chinese media.
# Optimized for high-performance parallel processing on multi-core systems.

import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import re
import jieba
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from tqdm import tqdm
import pickle
from datetime import datetime
import multiprocessing as mp
from functools import partial
import math
import time

# Set NLTK data path to a directory where we have write permission
nltk_data_dir = "/U_PZL2021KF0012/hx/EPF/History_and_digital/nltk_data"
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_dir)

# Set paths for Linux environment
DATA_PATH = "/U_PZL2021KF0012/hx/EPF/History_and_digital/project_data/processed_data"
OUTPUT_PATH = "/U_PZL2021KF0012/hx/EPF/History_and_digital/project_data/nationalism_analysis"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Get maximum available CPUs for parallel processing
MAX_CPUS = min(mp.cpu_count(), 100)  # Using up to 100 cores as mentioned
print(f"Setting up parallel processing with {MAX_CPUS} cores")

print("Starting Nationalist Language Analysis...")

# -----------------------------------------------------------------------
# 1. Load and prepare the datasets
# -----------------------------------------------------------------------

# Load standardized datasets
nyt_path = os.path.join(DATA_PATH, "nyt_standardized.pkl")
pd_path = os.path.join(DATA_PATH, "people_daily_standardized.pkl")

print("Loading datasets...")
nyt_df = pd.read_pickle(nyt_path)
pd_df = pd.read_pickle(pd_path)

print(f"NYT dataset: {len(nyt_df)} articles")
print(f"People's Daily dataset: {len(pd_df)} articles")

# Define Olympic years for later analysis
summer_olympic_years = [1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012]
winter_olympic_years = [1980, 1984, 1988, 1992, 1994, 1998, 2002, 2006, 2010, 2014]
olympic_years = sorted(list(set(summer_olympic_years + winter_olympic_years)))

# Mark whether articles are from Olympic years
nyt_df['is_olympic_year'] = nyt_df['year'].isin(olympic_years)
pd_df['is_olympic_year'] = pd_df['year'].isin(olympic_years)

# Create shared dictionaries for both languages upfront to improve performance
# -----------------------------------------------------------------------
# 2. Define Nationalist Vocabulary Dictionaries
# -----------------------------------------------------------------------

# English nationalist vocabulary - expanded for better coverage
english_nationalist_dict = {
    # National identity words
    'national_identity': [
        'america', 'american', 'americans', 'united states', 'usa', 'u.s.', 'u.s.a.', 
        'nation', 'national', 'country', 'homeland', 'fatherland', 'motherland', 
        'stars and stripes', 'uncle sam', 'yankee', 'yankees', 'stateside', 'home soil'
    ],
    
    # Patriotic emotion words
    'patriotic_emotion': [
        'pride', 'proud', 'proudly', 'patriotic', 'patriotism', 'honor', 'honour', 
        'glory', 'glorious', 'represent', 'representing', 'dignified', 'dignity', 'heritage',
        'loyalty', 'loyal', 'devotion', 'devoted', 'allegiance', 'patriot', 'patriots',
        'salute', 'tribute', 'homage', 'revere', 'reverence', 'cherish'
    ],
    
    # Collective belonging words
    'collective_belonging': [
        'we', 'our', 'us', 'ourselves', 'team usa', 'american team', 'u.s. team',
        'american athletes', 'u.s. athletes', 'american delegation', 'team america',
        'american representatives', 'countrymen', 'compatriot', 'compatriots',
        'fellow americans', 'nationals', 'citizens', 'citizenship', 'national team'
    ],
    
    # National symbols
    'national_symbols': [
        'flag', 'flags', 'anthem', 'star-spangled banner', 'stars and stripes',
        'red white and blue', 'red-white-and-blue', 'american flag', 'u.s. flag',
        'old glory', 'star spangled', 'stars & stripes', 'national emblem',
        'colors', 'colours', 'insignia', 'banner', 'eagle', 'american eagle', 'bald eagle'
    ],
    
    # National superiority words
    'national_superiority': [
        'best', 'greatest', 'superior', 'dominance', 'dominant', 'powerhouse', 
        'leader', 'leading', 'premier', 'preeminent', 'preeminent', 'superpower',
        'supremacy', 'supreme', 'exceptional', 'exceptionalism', 'triumph', 'triumphant',
        'victorious', 'unrivaled', 'unmatched', 'invincible', 'greatness', 'prestige',
        'mighty', 'might', 'powerful', 'strong', 'strength', 'excel', 'excellence',
        'outperform', 'outclass', 'outshine', 'outdo', 'surpass', 'prevail'
    ]
}

# Chinese nationalist vocabulary - expanded for better coverage
chinese_nationalist_dict = {
    # National identity words
    'national_identity': [
        '中国', '中华', '国家', '祖国', '我国', '国人', '中华人民共和国', '中华民族',
        '炎黄子孙', '龙的传人', '国土', '中国人', '华人', '国内', '本国', '国土',
        '华夏', '神州', '华夏大地', '神州大地'
    ],
    
    # Patriotic emotion words
    'patriotic_emotion': [
        '自豪', '骄傲', '荣耀', '光荣', '代表', '为国争光', '爱国', '忠诚', '热爱',
        '敬爱', '尊严', '自尊', '敬仰', '敬畏', '崇高', '神圣', '神气', '自信',
        '风采', '风貌', '精神', '爱国主义', '忠心', '自信心', '自豪感', '荣誉感',
        '奉献', '牺牲', '无私', '传统', '传承'
    ],
    
    # Collective belonging words
    'collective_belonging': [
        '我们', '我们的', '中国队', '国家队', '中国运动员', '中国代表团', '国人',
        '同胞', '中华儿女', '炎黄子孙', '国家的', '祖国的', '我国的', '中华民族的',
        '全国', '全国人民', '亿万人民', '十几亿', '十四亿', '共同', '集体', '团队',
        '人民', '民族', '中华大家庭'
    ],
    
    # National symbols
    'national_symbols': [
        '国旗', '国歌', '五星红旗', '红色', '国徽', '人民英雄纪念碑', '天安门',
        '长城', '长江', '黄河', '华表', '中华民族', '龙', '中国龙', '华表',
        '人民币', '人民大会堂', '中南海', '紫禁城', '故宫', '颜色', '红黄',
        '红色', '黄色', '五星', '象征', '标志', '旗帜'
    ],
    
    # National superiority words
    'national_superiority': [
        '最强', '最好', '强大', '领先', '超越', '引领', '第一', '冠军', '金牌',
        '夺冠', '胜利', '成功', '成就', '辉煌', '卓越', '杰出', '优秀', '优越',
        '先进', '领导', '领袖', '领航', '突破', '创新', '开创', '创造', '造就',
        '威望', '威严', '威武', '雄壮', '雄伟', '伟大', '壮丽', '壮观', '优越',
        '优势', '超过', '超越', '超强', '超级', '一流', '世界级', '世界一流',
        '霸主', '霸气', '霸道', '统治', '主宰', '主导', '支配', '控制', '完胜',
        '摧枯拉朽', '所向披靡', '势不可挡', '锐不可当', '所向无敌'
    ]
}

# Pre-initialize Jieba to avoid concurrency issues
print("Initializing Jieba Chinese segmentation...")
jieba.initialize()

# -----------------------------------------------------------------------
# 3. Text Processing Functions
# -----------------------------------------------------------------------

# Get English stopwords
english_stopwords = set(stopwords.words('english'))

# Define custom Chinese stopwords (a basic set)
chinese_stopwords = set([
    '的', '了', '和', '是', '在', '我', '有', '这', '他', '们', '到', '说', '就', '也',
    '着', '那', '与', '以', '很', '不', '为', '人', '都', '个', '来', '去', '还', '对',
    '啊', '吧', '呢', '吗', '嗯', '如', '从', '被', '又', '给', '可', '但', '能', '或',
    '把', '这个', '那个', '这样', '那样', '这些', '那些', '它', '她', '地', '上', '中',
    '下', '前', '后', '里', '外'
])

def process_english_text(text):
    """Process English text for analysis."""
    if not isinstance(text, str):
        return []
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and non-alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and word not in english_stopwords]
    
    return tokens

def process_chinese_text(text):
    """Process Chinese text for analysis."""
    if not isinstance(text, str):
        return []
    
    # Use jieba for Chinese word segmentation
    tokens = jieba.lcut(text)
    
    # Remove stopwords and single character tokens (often not meaningful)
    tokens = [word for word in tokens if word not in chinese_stopwords and len(word) > 1]
    
    return tokens

def count_nationalist_words(tokens, nationalist_dict, text=None):
    """Count occurrences of nationalist words in processed text tokens."""
    counts = {category: 0 for category in nationalist_dict.keys()}
    total_nationalist_words = 0
    
    # For efficiency, combine tokens into a single string for multi-word phrase search
    if text is None:
        text = ' '.join(tokens).lower()
    
    # Search for each category of nationalist words
    for category, word_list in nationalist_dict.items():
        for word in word_list:
            # For single words
            if ' ' not in word:
                counts[category] += tokens.count(word)
            # For phrases
            else:
                counts[category] += text.count(word)
    
    # Calculate total nationalist words
    total_nationalist_words = sum(counts.values())
    counts['total_nationalist_words'] = total_nationalist_words
    
    return counts

# -----------------------------------------------------------------------
# 4. Parallel Processing Functions
# -----------------------------------------------------------------------

def process_nyt_article_batch(articles_batch, nationalist_dict=english_nationalist_dict):
    """Process a batch of NYT articles in parallel."""
    results = []
    
    for _, article in articles_batch.iterrows():
        try:
            # Process text
            tokens = process_english_text(article['content'])
            token_count = len(tokens)
            
            if token_count == 0:
                continue
                
            # Count nationalist words
            text = ' '.join(tokens).lower()
            nationalist_counts = count_nationalist_words(tokens, nationalist_dict, text)
            
            # Calculate density (per 1000 words)
            nationalist_density = {
                f"{k}_density": (v / token_count) * 1000 
                for k, v in nationalist_counts.items()
            }
            
            # Combine results
            result = {
                'article_id': article['original_id'],
                'year': article['year'],
                'month': article['month'],
                'is_olympic_year': article['is_olympic_year'],
                'token_count': token_count,
                **nationalist_counts,
                **nationalist_density
            }
            results.append(result)
        except Exception as e:
            print(f"Error processing NYT article {article['original_id']}: {e}")
    
    return results

def process_peoples_daily_article_batch(articles_batch, nationalist_dict=chinese_nationalist_dict):
    """Process a batch of People's Daily articles in parallel."""
    results = []
    
    for _, article in articles_batch.iterrows():
        try:
            # Process text
            tokens = process_chinese_text(article['content'])
            token_count = len(tokens)
            
            if token_count == 0:
                continue
                
            # Count nationalist words
            text = ''.join(tokens)  # Chinese doesn't need spaces
            nationalist_counts = count_nationalist_words(tokens, nationalist_dict, text)
            
            # Calculate density (per 1000 words)
            nationalist_density = {
                f"{k}_density": (v / token_count) * 1000 
                for k, v in nationalist_counts.items()
            }
            
            # Combine results
            result = {
                'article_id': article['original_id'],
                'year': article['year'],
                'month': article['month'],
                'is_olympic_year': article['is_olympic_year'],
                'token_count': token_count,
                **nationalist_counts,
                **nationalist_density
            }
            results.append(result)
        except Exception as e:
            print(f"Error processing People's Daily article {article['original_id']}: {e}")
    
    return results

def analyze_nyt_articles_parallel(sample_size=None):
    """Analyze nationalist language in NYT articles using parallel processing."""
    print("\nAnalyzing NYT articles for nationalist language in parallel...")
    start_time = time.time()
    
    # Sample or use all data
    if sample_size and sample_size < len(nyt_df):
        articles = nyt_df.sample(sample_size, random_state=42)
    else:
        articles = nyt_df
    
    num_articles = len(articles)
    print(f"Processing {num_articles} NYT articles across {MAX_CPUS} cores")
    
    # Split the data into chunks for parallel processing
    batch_size = math.ceil(num_articles / MAX_CPUS)
    batches = [articles.iloc[i:i+batch_size] for i in range(0, num_articles, batch_size)]
    print(f"Created {len(batches)} batches, each with approximately {batch_size} articles")
    
    # Create a process pool
    with mp.Pool(processes=MAX_CPUS) as pool:
        # Process each batch in parallel
        batch_results = list(tqdm(
            pool.imap(process_nyt_article_batch, batches),
            total=len(batches),
            desc="Processing NYT batches"
        ))
    
    # Flatten the results
    results = [item for sublist in batch_results for item in sublist]
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    elapsed_time = time.time() - start_time
    print(f"Completed analysis of {len(results_df)} NYT articles in {elapsed_time:.2f} seconds")
    return results_df

def analyze_peoples_daily_articles_parallel(sample_size=None):
    """Analyze nationalist language in People's Daily articles using parallel processing."""
    print("\nAnalyzing People's Daily articles for nationalist language in parallel...")
    start_time = time.time()
    
    # Sample or use all data
    if sample_size and sample_size < len(pd_df):
        articles = pd_df.sample(sample_size, random_state=42)
    else:
        articles = pd_df
    
    num_articles = len(articles)
    print(f"Processing {num_articles} People's Daily articles across {MAX_CPUS} cores")
    
    # Split the data into chunks for parallel processing
    batch_size = math.ceil(num_articles / MAX_CPUS)
    batches = [articles.iloc[i:i+batch_size] for i in range(0, num_articles, batch_size)]
    print(f"Created {len(batches)} batches, each with approximately {batch_size} articles")
    
    # Create a process pool
    with mp.Pool(processes=MAX_CPUS) as pool:
        # Process each batch in parallel
        batch_results = list(tqdm(
            pool.imap(process_peoples_daily_article_batch, batches),
            total=len(batches),
            desc="Processing People's Daily batches"
        ))
    
    # Flatten the results
    results = [item for sublist in batch_results for item in sublist]
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    elapsed_time = time.time() - start_time
    print(f"Completed analysis of {len(results_df)} People's Daily articles in {elapsed_time:.2f} seconds")
    return results_df

# -----------------------------------------------------------------------
# 5. Visualization Functions
# -----------------------------------------------------------------------

def plot_nationalist_language_over_time(nyt_results, pd_results):
    """Plot nationalist language trends over time for both sources."""
    print("\nGenerating time series visualizations...")
    
    # Prepare data - group by year and calculate mean densities
    nyt_yearly = nyt_results.groupby('year')['total_nationalist_words_density'].mean().reset_index()
    pd_yearly = pd_results.groupby('year')['total_nationalist_words_density'].mean().reset_index()
    
    # Merge the data
    yearly_data = pd.merge(nyt_yearly, pd_yearly, on='year', suffixes=('_nyt', '_pd'))
    
    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(yearly_data['year'], yearly_data['total_nationalist_words_density_nyt'], 
             label='NYT', marker='o', linewidth=2, color='#1f77b4')
    plt.plot(yearly_data['year'], yearly_data['total_nationalist_words_density_pd'], 
             label='People\'s Daily', marker='s', linewidth=2, color='#ff7f0e')
    
    # Add vertical lines for Olympic years
    for year in olympic_years:
        plt.axvline(x=year, color='gray', linestyle='--', alpha=0.5)
    
    # Add vertical line for 2008 Beijing Olympics with special styling
    plt.axvline(x=2008, color='red', linestyle='-', alpha=0.7)
    plt.text(2008, plt.ylim()[1]*0.95, 'Beijing\nOlympics', 
             ha='center', va='top', color='red', fontweight='bold')
    
    # Formatting
    plt.title('Nationalist Language Density Over Time (1980-2015)', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Nationalist Words per 1000 Words', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(olympic_years, rotation=45)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'nationalist_language_time_trend.png'), dpi=300)
    plt.close()

def plot_nationalist_categories_comparison(nyt_results, pd_results):
    """Plot comparison of nationalist language categories between the two sources."""
    print("Generating category comparison visualizations...")
    
    # Get average densities for each category
    categories = list(english_nationalist_dict.keys())
    
    nyt_means = [nyt_results[f'{cat}_density'].mean() for cat in categories]
    pd_means = [pd_results[f'{cat}_density'].mean() for cat in categories]
    
    # Create more readable category labels
    category_labels = [' '.join(cat.split('_')).title() for cat in categories]
    
    # Create bar chart
    plt.figure(figsize=(12, 8))
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, nyt_means, width, label='NYT', color='#1f77b4')
    plt.bar(x + width/2, pd_means, width, label='People\'s Daily', color='#ff7f0e')
    
    plt.title('Comparison of Nationalist Language Categories', fontsize=16)
    plt.ylabel('Words per 1000 Words', fontsize=12)
    plt.xticks(x, category_labels, rotation=45, ha='right')
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'nationalist_categories_comparison.png'), dpi=300)
    plt.close()

def plot_olympic_vs_non_olympic(nyt_results, pd_results):
    """Compare nationalist language between Olympic and non-Olympic years."""
    print("Generating Olympic vs non-Olympic years comparison...")
    
    # Calculate mean densities for Olympic and non-Olympic years
    nyt_olympic = nyt_results[nyt_results['is_olympic_year']]['total_nationalist_words_density'].mean()
    nyt_non_olympic = nyt_results[~nyt_results['is_olympic_year']]['total_nationalist_words_density'].mean()
    
    pd_olympic = pd_results[pd_results['is_olympic_year']]['total_nationalist_words_density'].mean()
    pd_non_olympic = pd_results[~pd_results['is_olympic_year']]['total_nationalist_words_density'].mean()
    
    # Create grouped bar chart
    plt.figure(figsize=(10, 6))
    x = np.arange(2)
    width = 0.35
    
    plt.bar(x - width/2, [nyt_olympic, nyt_non_olympic], width, label='NYT', color='#1f77b4')
    plt.bar(x + width/2, [pd_olympic, pd_non_olympic], width, label='People\'s Daily', color='#ff7f0e')
    
    plt.title('Nationalist Language: Olympic vs. Non-Olympic Years', fontsize=16)
    plt.ylabel('Nationalist Words per 1000 Words', fontsize=12)
    plt.xticks(x, ['Olympic Years', 'Non-Olympic Years'])
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'olympic_vs_non_olympic_comparison.png'), dpi=300)
    plt.close()

def plot_beijing_olympics_impact(nyt_results, pd_results):
    """Analyze and visualize the impact of the 2008 Beijing Olympics."""
    print("Analyzing the impact of 2008 Beijing Olympics...")
    
    # Define periods for analysis
    pre_beijing = [2004, 2005, 2006, 2007]  # Pre-Beijing Olympics
    beijing_year = [2008]                   # Beijing Olympics year
    post_beijing = [2009, 2010, 2011, 2012] # Post-Beijing Olympics
    
    # Filter results for these periods
    nyt_pre = nyt_results[nyt_results['year'].isin(pre_beijing)]
    nyt_during = nyt_results[nyt_results['year'].isin(beijing_year)]
    nyt_post = nyt_results[nyt_results['year'].isin(post_beijing)]
    
    pd_pre = pd_results[pd_results['year'].isin(pre_beijing)]
    pd_during = pd_results[pd_results['year'].isin(beijing_year)]
    pd_post = pd_results[pd_results['year'].isin(post_beijing)]
    
    # Calculate mean nationalist language density for each period
    nyt_values = [
        nyt_pre['total_nationalist_words_density'].mean(),
        nyt_during['total_nationalist_words_density'].mean(),
        nyt_post['total_nationalist_words_density'].mean()
    ]
    
    pd_values = [
        pd_pre['total_nationalist_words_density'].mean(),
        pd_during['total_nationalist_words_density'].mean(),
        pd_post['total_nationalist_words_density'].mean()
    ]
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    x = np.arange(3)
    width = 0.35
    
    plt.bar(x - width/2, nyt_values, width, label='NYT', color='#1f77b4')
    plt.bar(x + width/2, pd_values, width, label='People\'s Daily', color='#ff7f0e')
    
    plt.title('Impact of 2008 Beijing Olympics on Nationalist Language', fontsize=16)
    plt.ylabel('Nationalist Words per 1000 Words', fontsize=12)
    plt.xticks(x, ['2004-2007\n(Pre-Beijing)', '2008\n(Beijing Olympics)', '2009-2012\n(Post-Beijing)'])
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'beijing_olympics_impact.png'), dpi=300)
    plt.close()

def plot_monthly_patterns(nyt_results, pd_results):
    """Visualize monthly patterns of nationalist language for Olympic years."""
    print("Generating monthly patterns visualization for Olympic years...")
    
    # Filter for Olympic years
    nyt_olympic = nyt_results[nyt_results['is_olympic_year']]
    pd_olympic = pd_results[pd_results['is_olympic_year']]
    
    # Group by month and calculate means
    nyt_monthly = nyt_olympic.groupby('month')['total_nationalist_words_density'].mean().reset_index()
    pd_monthly = pd_olympic.groupby('month')['total_nationalist_words_density'].mean().reset_index()
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(nyt_monthly['month'], nyt_monthly['total_nationalist_words_density'], 
             marker='o', linewidth=2, label='NYT', color='#1f77b4')
    plt.plot(pd_monthly['month'], pd_monthly['total_nationalist_words_density'], 
             marker='s', linewidth=2, label='People\'s Daily', color='#ff7f0e')
    
    # Highlight summer and winter Olympics typical months
    plt.axvspan(1.5, 3.5, alpha=0.2, color='lightblue', label='Winter Olympics (Feb-Mar)')
    plt.axvspan(6.5, 8.5, alpha=0.2, color='lightyellow', label='Summer Olympics (Jul-Aug)')
    
    plt.title('Monthly Patterns of Nationalist Language in Olympic Years', fontsize=16)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Nationalist Words per 1000 Words', fontsize=12)
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'monthly_nationalist_patterns.png'), dpi=300)
    plt.close()

# -----------------------------------------------------------------------
# 6. Main Execution
# -----------------------------------------------------------------------

def main():
    """Main execution function."""
    start_time = time.time()
    
    # Print dictionaries for verification
    print("\nEnglish Nationalist Dictionary:")
    for category, words in english_nationalist_dict.items():
        print(f"  {category}: {len(words)} terms")
    
    print("\nChinese Nationalist Dictionary:")
    for category, words in chinese_nationalist_dict.items():
        print(f"  {category}: {len(words)} terms")
    
    # For initial development, use a smaller sample size to test the code
    # You can adjust these values or set to None to process all articles
    nyt_sample_size = None  # Set to None to process all articles
    pd_sample_size = None   # Set to None to process all articles
    
    # Run analysis for NYT
    nyt_results_path = os.path.join(OUTPUT_PATH, "nyt_nationalist_results.pkl")
    if os.path.exists(nyt_results_path):
        print(f"\nLoading existing NYT results from {nyt_results_path}")
        nyt_results = pd.read_pickle(nyt_results_path)
    else:
        nyt_results = analyze_nyt_articles_parallel(nyt_sample_size)
        nyt_results.to_pickle(nyt_results_path)
        print(f"Saved NYT results to {nyt_results_path}")
    
    # Run analysis for People's Daily
    pd_results_path = os.path.join(OUTPUT_PATH, "pd_nationalist_results.pkl")
    if os.path.exists(pd_results_path):
        print(f"\nLoading existing People's Daily results from {pd_results_path}")
        pd_results = pd.read_pickle(pd_results_path)
    else:
        pd_results = analyze_peoples_daily_articles_parallel(pd_sample_size)
        pd_results.to_pickle(pd_results_path)
        print(f"Saved People's Daily results to {pd_results_path}")
    
    # Generate visualizations
    plot_nationalist_language_over_time(nyt_results, pd_results)
    plot_nationalist_categories_comparison(nyt_results, pd_results)
    plot_olympic_vs_non_olympic(nyt_results, pd_results)
    plot_beijing_olympics_impact(nyt_results, pd_results)
    plot_monthly_patterns(nyt_results, pd_results)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nNYT Nationalist Language (per 1000 words):")
    print(nyt_results[['total_nationalist_words_density'] + 
                      [f"{cat}_density" for cat in english_nationalist_dict.keys()]].describe())
    
    print("\nPeople's Daily Nationalist Language (per 1000 words):")
    print(pd_results[['total_nationalist_words_density'] + 
                     [f"{cat}_density" for cat in chinese_nationalist_dict.keys()]].describe())
    
    total_time = time.time() - start_time
    print(f"\nTotal analysis time: {total_time:.2f} seconds")
    print("\nAnalysis complete! Visualizations saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    # Use 'spawn' method for more reliable multiprocessing in large workloads
    mp.set_start_method('spawn', force=True)
    main()