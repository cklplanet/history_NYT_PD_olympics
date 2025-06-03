# Project 2: Medal Mention Analysis & Victory Definition Differences
# =================================================================
# This script analyzes how NYT and People's Daily differently mention and frame
# Olympic medals, investigating potential differences in how "victory" is defined
# (gold medals vs total medals) in American and Chinese media.
# Optimized for high-performance parallel processing on multi-core systems.

import os
import pandas as pd
import numpy as np
import re
import jieba
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
from datetime import datetime
import multiprocessing as mp
from functools import partial
import math
import time
from wordcloud import WordCloud
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

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
OUTPUT_PATH = "/U_PZL2021KF0012/hx/EPF/History_and_digital/project_data/medal_analysis"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Get maximum available CPUs for parallel processing
MAX_CPUS = min(mp.cpu_count(), 100)  # Using up to 100 cores as mentioned
print(f"Setting up parallel processing with {MAX_CPUS} cores")

print("Starting Medal Mention Analysis...")

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

# -----------------------------------------------------------------------
# 2. Define Medal-related Vocabulary Dictionaries
# -----------------------------------------------------------------------

# English medal-related vocabulary
english_medal_dict = {
    # Gold medal related words
    'gold_medal': [
        'gold medal', 'gold medals', 'gold medalist', 'gold medalists', 'gold medallist', 'gold medallists',
        'first place', 'first-place', '1st place', 'gold-medal', 'olympic champion', 'olympic gold',
        'gold-winning', 'gold winner', 'olympic title', 'olympic championship'
    ],
    
    # Silver medal related words
    'silver_medal': [
        'silver medal', 'silver medals', 'silver medalist', 'silver medalists', 'silver medallist', 'silver medallists',
        'second place', 'second-place', '2nd place', 'silver-medal', 'olympic silver',
        'silver-winning', 'silver winner', 'runner-up', 'runner up'
    ],
    
    # Bronze medal related words
    'bronze_medal': [
        'bronze medal', 'bronze medals', 'bronze medalist', 'bronze medalists', 'bronze medallist', 'bronze medallists',
        'third place', 'third-place', '3rd place', 'bronze-medal', 'olympic bronze',
        'bronze-winning', 'bronze winner'
    ],
    
    # General medal or ranking words
    'general_medal': [
        'medal count', 'medal table', 'medal tally', 'medal standings', 'medal totals',
        'podium', 'medal race', 'medal haul', 'top three', 'winning medal',
        'total medals', 'overall medals', 'medal winner', 'medal winners'
    ]
}

# Chinese medal-related vocabulary
chinese_medal_dict = {
    # Gold medal related words
    'gold_medal': [
        '金牌', '冠军', '第一名', '第一位', '金牌得主', '夺冠', '获得冠军', 
        '奥运冠军', '夺得金牌', '获得金牌', '拿下金牌', '摘得金牌', '登顶',
        '第1名', '头名', '实现金牌零的突破', '获胜', '夺魁'
    ],
    
    # Silver medal related words
    'silver_medal': [
        '银牌', '亚军', '第二名', '第二位', '银牌得主', '获得亚军', 
        '夺得银牌', '获得银牌', '拿下银牌', '摘得银牌',
        '第2名', '屈居亚军', '亚军获得者'
    ],
    
    # Bronze medal related words
    'bronze_medal': [
        '铜牌', '季军', '第三名', '第三位', '铜牌得主', '获得季军', 
        '夺得铜牌', '获得铜牌', '拿下铜牌', '摘得铜牌',
        '第3名', '铜牌获得者'
    ],
    
    # General medal or ranking words
    'general_medal': [
        '奖牌', '奖牌得主', '奖牌榜', '奖牌数', '奖牌总数', '总奖牌', '奖牌积分', 
        '领奖台', '颁奖台', '前三名', '奖牌获得者', '奖牌成绩', '总奖牌数',
        '奖牌统计', '排行榜', '奖牌榜排名', '奖牌总榜', '金银铜'
    ]
}

# List of major countries for country-medal association analysis
english_countries = {
    'usa': ['united states', 'usa', 'u.s.', 'u.s.a.', 'american', 'americans', 'america', 'team usa'],
    'china': ['china', 'chinese', 'prc', 'people\'s republic of china', 'team china'],
    'russia': ['russia', 'russian', 'russians', 'soviet union', 'soviet', 'ussr', 'team russia', 'russian federation'],
    'germany': ['germany', 'german', 'germans', 'west germany', 'east germany', 'gdr', 'frg', 'team germany'],
    'japan': ['japan', 'japanese', 'team japan'],
    'uk': ['great britain', 'united kingdom', 'uk', 'british', 'england', 'team gb', 'britain'],
    'france': ['france', 'french', 'team france'],
    'australia': ['australia', 'australian', 'australians', 'team australia'],
    'canada': ['canada', 'canadian', 'canadians', 'team canada'],
    'italy': ['italy', 'italian', 'italians', 'team italy']
}

chinese_countries = {
    'usa': ['美国', '美国队', '美国人', '美利坚', '美'],
    'china': ['中国', '中国队', '国家队', '中华', '国人', '我国', '我们', '中'],
    'russia': ['俄罗斯', '俄罗斯队', '俄国', '俄', '苏联', '苏联队', '苏'],
    'germany': ['德国', '德国队', '德', '西德', '东德', '联邦德国', '民主德国'],
    'japan': ['日本', '日本队', '日', '日本国'],
    'uk': ['英国', '英国队', '英', '大不列颠', '不列颠', '联合王国'],
    'france': ['法国', '法国队', '法'],
    'australia': ['澳大利亚', '澳大利亚队', '澳大利亚国家队', '澳'],
    'canada': ['加拿大', '加拿大队', '加'],
    'italy': ['意大利', '意大利队', '意']
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

def preprocess_english_text(text):
    """Preprocess English text for medal mention analysis."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace line breaks with spaces
    text = re.sub(r'\n', ' ', text)
    
    return text

def preprocess_chinese_text(text):
    """Preprocess Chinese text for medal mention analysis."""
    if not isinstance(text, str):
        return ""
    
    # Replace line breaks with spaces
    text = re.sub(r'\n', ' ', text)
    
    return text

def find_medal_mentions(text, medal_dict, is_english=True):
    """Find medal mentions in text and return counts by category."""
    if not text:
        return {'gold_medal': 0, 'silver_medal': 0, 'bronze_medal': 0, 'general_medal': 0, 'total_medal_mentions': 0}
    
    counts = {'gold_medal': 0, 'silver_medal': 0, 'bronze_medal': 0, 'general_medal': 0}
    
    # Detect medal mentions by category
    for category, terms in medal_dict.items():
        for term in terms:
            # Count occurrences of each term
            term_count = text.count(term)
            counts[category] += term_count
    
    # Calculate total medal mentions
    counts['total_medal_mentions'] = sum(counts.values())
    
    return counts

def find_country_mentions(text, country_dict):
    """Find country mentions in text and return counts by country."""
    if not text:
        return {country: 0 for country in country_dict}, 0
    
    counts = {country: 0 for country in country_dict}
    
    # Detect country mentions
    for country, terms in country_dict.items():
        for term in terms:
            # Count occurrences of each term
            term_count = text.count(term)
            counts[country] += term_count
    
    # Calculate total country mentions
    total_country_mentions = sum(counts.values())
    
    return counts, total_country_mentions

def get_context_windows(text, target_words, window_size=5, is_english=True):
    """Extract context windows around medal mentions."""
    if not text or not target_words:
        return []
    
    context_windows = []
    
    if is_english:
        # Split text into sentences for English
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            for i, word in enumerate(words):
                # Check for single-word target
                if word in target_words:
                    start = max(0, i - window_size)
                    end = min(len(words), i + window_size + 1)
                    context = ' '.join(words[start:end])
                    context_windows.append(context)
                
                # Check for multi-word targets
                if i < len(words) - 1:
                    bigram = word + ' ' + words[i+1]
                    if bigram in target_words:
                        start = max(0, i - window_size)
                        end = min(len(words), i + window_size + 2)
                        context = ' '.join(words[start:end])
                        context_windows.append(context)
    else:
        # For Chinese, we use a simple sliding window approach
        # First, we segment the text
        words = jieba.lcut(text)
        
        for i, word in enumerate(words):
            # Check if current word is in target_words
            if word in target_words:
                start = max(0, i - window_size)
                end = min(len(words), i + window_size + 1)
                context = ''.join(words[start:end])
                context_windows.append(context)
    
    return context_windows

def find_country_medal_associations(text, medal_dict, country_dict, window_size=30, is_english=True):
    """Find associations between countries and medal types within context windows."""
    if not text:
        return {}
    
    # Initialize result dictionary
    associations = {
        country: {medal_type: 0 for medal_type in medal_dict} 
        for country in country_dict
    }
    
    # Flatten medal terms lists for faster lookup
    flat_medal_terms = {}
    for medal_type, terms in medal_dict.items():
        for term in terms:
            flat_medal_terms[term] = medal_type
    
    # Flatten country terms lists for faster lookup
    flat_country_terms = {}
    for country, terms in country_dict.items():
        for term in terms:
            flat_country_terms[term] = country
    
    if is_english:
        # Process English text by sentences
        sentences = sent_tokenize(text.lower())
        for sentence in sentences:
            # Analyze association only if sentence is not too long
            # Very long sentences might not represent genuine associations
            if len(sentence) <= 500:  # Arbitrary length limit
                # Check each medal term
                for medal_term, medal_type in flat_medal_terms.items():
                    if medal_term in sentence:
                        # If medal term found, check for country mentions in same sentence
                        for country_term, country in flat_country_terms.items():
                            if country_term in sentence:
                                associations[country][medal_type] += 1
    else:
        # Process Chinese text
        # Since Chinese doesn't have clear sentence boundaries like English,
        # we'll use punctuation as delimiters
        segments = re.split(r'[。！？；]', text)
        for segment in segments:
            if len(segment) <= 200:  # Arbitrary length limit
                # Check each medal term
                for medal_term, medal_type in flat_medal_terms.items():
                    if medal_term in segment:
                        # If medal term found, check for country mentions in same segment
                        for country_term, country in flat_country_terms.items():
                            if country_term in segment:
                                associations[country][medal_type] += 1
    
    return associations

# -----------------------------------------------------------------------
# 4. Parallel Processing Functions
# -----------------------------------------------------------------------

def process_nyt_article_batch(articles_batch):
    """Process a batch of NYT articles in parallel."""
    results = []
    
    for _, article in articles_batch.iterrows():
        try:
            # Get article content
            content = article['content']
            if not isinstance(content, str) or not content.strip():
                continue
            
            # Preprocess text
            processed_text = preprocess_english_text(content)
            word_count = len(processed_text.split())  # Count words for English text
            
            # Count medal mentions
            medal_counts = find_medal_mentions(processed_text, english_medal_dict, is_english=True)
            
            # Count country mentions
            country_counts, total_country_mentions = find_country_mentions(processed_text, english_countries)
            
            # Analyze country-medal associations
            country_medal_assoc = find_country_medal_associations(
                processed_text, english_medal_dict, english_countries, is_english=True
            )
            
            # Get context windows for medal mentions (focusing on gold and general medals)
            gold_contexts = get_context_windows(
                processed_text, english_medal_dict['gold_medal'], window_size=5, is_english=True
            )
            general_medal_contexts = get_context_windows(
                processed_text, english_medal_dict['general_medal'], window_size=5, is_english=True
            )
            
            # Calculate metrics
            # Gold to total medal mentions ratio
            if medal_counts['total_medal_mentions'] > 0:
                gold_ratio = medal_counts['gold_medal'] / medal_counts['total_medal_mentions']
            else:
                gold_ratio = 0
                
            # General to specific medal mentions ratio
            specific_medals = medal_counts['gold_medal'] + medal_counts['silver_medal'] + medal_counts['bronze_medal']
            if specific_medals + medal_counts['general_medal'] > 0:
                general_ratio = medal_counts['general_medal'] / (specific_medals + medal_counts['general_medal'])
            else:
                general_ratio = 0
            
            # Combine results
            result = {
                'article_id': article['original_id'],
                'year': article['year'],
                'month': article['month'],
                'is_olympic_year': article['is_olympic_year'],
                'content_length': len(processed_text),
                'word_count': word_count,
                **medal_counts,
                'gold_ratio': gold_ratio,
                'general_ratio': general_ratio,
                'country_mentions': country_counts,
                'total_country_mentions': total_country_mentions,
                'country_medal_assoc': country_medal_assoc,
                'gold_contexts': gold_contexts,
                'general_medal_contexts': general_medal_contexts
            }
            results.append(result)
        except Exception as e:
            print(f"Error processing NYT article {article['original_id']}: {e}")
    
    return results

def process_peoples_daily_article_batch(articles_batch):
    """Process a batch of People's Daily articles in parallel."""
    results = []
    
    for _, article in articles_batch.iterrows():
        try:
            # Get article content
            content = article['content']
            if not isinstance(content, str) or not content.strip():
                continue
            
            # Preprocess text
            processed_text = preprocess_chinese_text(content)
            word_count = len(jieba.lcut(processed_text))  # Count words for Chinese text using jieba
            
            # Count medal mentions
            medal_counts = find_medal_mentions(processed_text, chinese_medal_dict, is_english=False)
            
            # Count country mentions
            country_counts, total_country_mentions = find_country_mentions(processed_text, chinese_countries)
            
            # Analyze country-medal associations
            country_medal_assoc = find_country_medal_associations(
                processed_text, chinese_medal_dict, chinese_countries, is_english=False
            )
            
            # Get context windows for medal mentions (focusing on gold and general medals)
            gold_contexts = get_context_windows(
                processed_text, chinese_medal_dict['gold_medal'], window_size=5, is_english=False
            )
            general_medal_contexts = get_context_windows(
                processed_text, chinese_medal_dict['general_medal'], window_size=5, is_english=False
            )
            
            # Calculate metrics
            # Gold to total medal mentions ratio
            if medal_counts['total_medal_mentions'] > 0:
                gold_ratio = medal_counts['gold_medal'] / medal_counts['total_medal_mentions']
            else:
                gold_ratio = 0
                
            # General to specific medal mentions ratio
            specific_medals = medal_counts['gold_medal'] + medal_counts['silver_medal'] + medal_counts['bronze_medal']
            if specific_medals + medal_counts['general_medal'] > 0:
                general_ratio = medal_counts['general_medal'] / (specific_medals + medal_counts['general_medal'])
            else:
                general_ratio = 0
            
            # Combine results
            result = {
                'article_id': article['original_id'],
                'year': article['year'],
                'month': article['month'],
                'is_olympic_year': article['is_olympic_year'],
                'content_length': len(processed_text),
                'word_count': word_count,
                **medal_counts,
                'gold_ratio': gold_ratio,
                'general_ratio': general_ratio,
                'country_mentions': country_counts,
                'total_country_mentions': total_country_mentions,
                'country_medal_assoc': country_medal_assoc,
                'gold_contexts': gold_contexts,
                'general_medal_contexts': general_medal_contexts
            }
            results.append(result)
        except Exception as e:
            print(f"Error processing People's Daily article {article['original_id']}: {e}")
    
    return results

def analyze_nyt_articles_parallel(sample_size=None):
    """Analyze medal mentions in NYT articles using parallel processing."""
    print("\nAnalyzing medal mentions in NYT articles in parallel...")
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
    
    # Process context data
    all_gold_contexts = []
    all_general_contexts = []
    
    for item in results:
        all_gold_contexts.extend(item['gold_contexts'])
        all_general_contexts.extend(item['general_medal_contexts'])
        
        # Remove contexts from individual items to save memory
        item.pop('gold_contexts')
        item.pop('general_medal_contexts')
    
    # Store contexts separately
    contexts = {
        'gold_contexts': all_gold_contexts,
        'general_medal_contexts': all_general_contexts
    }
    
    elapsed_time = time.time() - start_time
    print(f"Completed medal analysis of {len(results)} NYT articles in {elapsed_time:.2f} seconds")
    return results, contexts

def analyze_peoples_daily_articles_parallel(sample_size=None):
    """Analyze medal mentions in People's Daily articles using parallel processing."""
    print("\nAnalyzing medal mentions in People's Daily articles in parallel...")
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
    
    # Process context data
    all_gold_contexts = []
    all_general_contexts = []
    
    for item in results:
        all_gold_contexts.extend(item['gold_contexts'])
        all_general_contexts.extend(item['general_medal_contexts'])
        
        # Remove contexts from individual items to save memory
        item.pop('gold_contexts')
        item.pop('general_medal_contexts')
    
    # Store contexts separately
    contexts = {
        'gold_contexts': all_gold_contexts,
        'general_medal_contexts': all_general_contexts
    }
    
    elapsed_time = time.time() - start_time
    print(f"Completed medal analysis of {len(results)} People's Daily articles in {elapsed_time:.2f} seconds")
    return results, contexts

# -----------------------------------------------------------------------
# 5. Visualization Functions
# -----------------------------------------------------------------------

def plot_medal_type_comparison(nyt_results, pd_results):
    """Plot comparison of medal type mentions between the two sources."""
    print("\nGenerating medal type comparison visualization...")
    
    # Extract medal mention data
    nyt_data = pd.DataFrame(nyt_results)
    pd_data = pd.DataFrame(pd_results)
    
    # Calculate medal mentions per article
    nyt_gold_per_article = nyt_data['gold_medal'].sum() / len(nyt_data)
    nyt_silver_per_article = nyt_data['silver_medal'].sum() / len(nyt_data)
    nyt_bronze_per_article = nyt_data['bronze_medal'].sum() / len(nyt_data)
    nyt_general_per_article = nyt_data['general_medal'].sum() / len(nyt_data)
    
    pd_gold_per_article = pd_data['gold_medal'].sum() / len(pd_data)
    pd_silver_per_article = pd_data['silver_medal'].sum() / len(pd_data)
    pd_bronze_per_article = pd_data['bronze_medal'].sum() / len(pd_data)
    pd_general_per_article = pd_data['general_medal'].sum() / len(pd_data)
    
    # Create plot data
    medal_types = ['Gold Medal', 'Silver Medal', 'Bronze Medal', 'General Medal']
    nyt_values = [nyt_gold_per_article, nyt_silver_per_article, nyt_bronze_per_article, nyt_general_per_article]
    pd_values = [pd_gold_per_article, pd_silver_per_article, pd_bronze_per_article, pd_general_per_article]
    
    # Create bar chart
    plt.figure(figsize=(12, 7))
    x = np.arange(len(medal_types))
    width = 0.35
    
    plt.bar(x - width/2, nyt_values, width, label='NYT', color='#1f77b4')
    plt.bar(x + width/2, pd_values, width, label='People\'s Daily', color='#ff7f0e')
    
    plt.title('Average Medal Mentions per Article by Medal Type (1980-2015)', fontsize=16)
    plt.ylabel('Mentions per Article', fontsize=12)
    plt.xticks(x, medal_types, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on top of each bar
    for i, v in enumerate(nyt_values):
        plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
    
    for i, v in enumerate(pd_values):
        plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'medal_type_comparison.png'), dpi=300)
    plt.close()
    
    # Also create a pie chart showing proportion of medal types for each source
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # NYT Medal Type Proportion
    nyt_sizes = [nyt_data['gold_medal'].sum(), nyt_data['silver_medal'].sum(), 
                nyt_data['bronze_medal'].sum(), nyt_data['general_medal'].sum()]
    nyt_labels = [f'Gold: {nyt_sizes[0]/sum(nyt_sizes):.1%}', 
                 f'Silver: {nyt_sizes[1]/sum(nyt_sizes):.1%}',
                 f'Bronze: {nyt_sizes[2]/sum(nyt_sizes):.1%}', 
                 f'General: {nyt_sizes[3]/sum(nyt_sizes):.1%}']
    
    ax1.pie(nyt_sizes, labels=nyt_labels, autopct='', startangle=90, colors=['gold', 'silver', '#CD7F32', '#1f77b4'], textprops={'fontsize': 15})
    ax1.set_title('NYT Medal Type Distribution', fontsize=16)
    
    # People's Daily Medal Type Proportion
    pd_sizes = [pd_data['gold_medal'].sum(), pd_data['silver_medal'].sum(), 
               pd_data['bronze_medal'].sum(), pd_data['general_medal'].sum()]
    pd_labels = [f'Gold: {pd_sizes[0]/sum(pd_sizes):.1%}', 
                f'Silver: {pd_sizes[1]/sum(pd_sizes):.1%}',
                f'Bronze: {pd_sizes[2]/sum(pd_sizes):.1%}', 
                f'General: {pd_sizes[3]/sum(pd_sizes):.1%}']
    
    ax2.pie(pd_sizes, labels=pd_labels, autopct='', startangle=90, colors=['gold', 'silver', '#CD7F32', '#ff7f0e'], textprops={'fontsize': 15})
    ax2.set_title('People\'s Daily Medal Type Distribution', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'medal_type_distribution_pie.png'), dpi=300)
    plt.close()

def plot_gold_ratio_over_time(nyt_results, pd_results):
    """Plot the ratio of gold medal mentions to total medal mentions over time."""
    print("Generating gold ratio over time visualization...")
    
    # Convert to DataFrame
    nyt_df = pd.DataFrame(nyt_results)
    pd_df = pd.DataFrame(pd_results)
    
    # Group by year and calculate mean gold ratio
    nyt_yearly = nyt_df.groupby('year')['gold_ratio'].mean().reset_index()
    pd_yearly = pd_df.groupby('year')['gold_ratio'].mean().reset_index()
    
    # Merge the data
    yearly_data = pd.merge(nyt_yearly, pd_yearly, on='year', suffixes=('_nyt', '_pd'))
    
    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(yearly_data['year'], yearly_data['gold_ratio_nyt'], 
             label='NYT', marker='o', linewidth=2, color='#1f77b4')
    plt.plot(yearly_data['year'], yearly_data['gold_ratio_pd'], 
             label='People\'s Daily', marker='s', linewidth=2, color='#ff7f0e')
    
    # Add vertical lines for Olympic years
    for year in olympic_years:
        plt.axvline(x=year, color='gray', linestyle='--', alpha=0.5)
    
    # Add vertical line for 2008 Beijing Olympics with special styling
    plt.axvline(x=2008, color='red', linestyle='-', alpha=0.7)
    plt.text(2008, plt.ylim()[1]*0.95, 'Beijing\nOlympics', 
             ha='center', va='top', color='red', fontweight='bold')
    
    # Formatting
    plt.title('Gold Medal Focus: Ratio of Gold Medal Mentions to Total Medal Mentions (1980-2015)', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Gold Medal Mentions / Total Medal Mentions', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.xticks(olympic_years, rotation=45)
    
    # Set y-axis to percentage format
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'gold_ratio_over_time.png'), dpi=300)
    plt.close()

def plot_medal_mentions_by_country(nyt_results, pd_results):
    """Plot association between medal types and countries in both sources."""
    print("Generating medal mentions by country visualization...")
    
    # Extract country-medal association data
    nyt_country_medal_assoc = {}
    for result in nyt_results:
        assoc = result.get('country_medal_assoc', {})
        for country, medal_counts in assoc.items():
            if country not in nyt_country_medal_assoc:
                nyt_country_medal_assoc[country] = {'gold_medal': 0, 'silver_medal': 0, 'bronze_medal': 0, 'general_medal': 0}
            
            for medal_type, count in medal_counts.items():
                nyt_country_medal_assoc[country][medal_type] += count
    
    pd_country_medal_assoc = {}
    for result in pd_results:
        assoc = result.get('country_medal_assoc', {})
        for country, medal_counts in assoc.items():
            if country not in pd_country_medal_assoc:
                pd_country_medal_assoc[country] = {'gold_medal': 0, 'silver_medal': 0, 'bronze_medal': 0, 'general_medal': 0}
            
            for medal_type, count in medal_counts.items():
                pd_country_medal_assoc[country][medal_type] += count
    
    # Create DataFrames for plotting
    nyt_data = []
    for country, medal_counts in nyt_country_medal_assoc.items():
        for medal_type, count in medal_counts.items():
            nyt_data.append({'Country': country, 'Medal Type': medal_type, 'Count': count, 'Source': 'NYT'})
    
    pd_data = []
    for country, medal_counts in pd_country_medal_assoc.items():
        for medal_type, count in medal_counts.items():
            pd_data.append({'Country': country, 'Medal Type': medal_type, 'Count': count, 'Source': 'People\'s Daily'})
    
    # Combine data
    all_data = pd.DataFrame(nyt_data + pd_data)
    
    # Filter to include only the top 6 countries by total medal mentions
    country_totals = all_data.groupby('Country')['Count'].sum().reset_index()
    top_countries = country_totals.sort_values('Count', ascending=False).head(6)['Country'].tolist()
    
    filtered_data = all_data[all_data['Country'].isin(top_countries)]
    
    # Create stacked bar chart for NYT
    nyt_filtered = filtered_data[filtered_data['Source'] == 'NYT']
    nyt_pivot = nyt_filtered.pivot_table(index='Country', columns='Medal Type', values='Count')
    
    # Ensure consistent column order and replace NaNs with 0
    all_medal_types = ['gold_medal', 'silver_medal', 'bronze_medal', 'general_medal']
    for medal_type in all_medal_types:
        if medal_type not in nyt_pivot.columns:
            nyt_pivot[medal_type] = 0
    nyt_pivot = nyt_pivot[all_medal_types].fillna(0)
    
    # Sort by total medal mentions
    nyt_pivot['total'] = nyt_pivot.sum(axis=1)
    nyt_pivot = nyt_pivot.sort_values('total', ascending=False)
    nyt_pivot = nyt_pivot.drop('total', axis=1)
    
    # Create stacked bar chart for People's Daily
    pd_filtered = filtered_data[filtered_data['Source'] == 'People\'s Daily']
    pd_pivot = pd_filtered.pivot_table(index='Country', columns='Medal Type', values='Count')
    
    # Ensure consistent column order and replace NaNs with 0
    for medal_type in all_medal_types:
        if medal_type not in pd_pivot.columns:
            pd_pivot[medal_type] = 0
    pd_pivot = pd_pivot[all_medal_types].fillna(0)
    
    # Sort by total medal mentions
    pd_pivot['total'] = pd_pivot.sum(axis=1)
    pd_pivot = pd_pivot.sort_values('total', ascending=False)
    pd_pivot = pd_pivot.drop('total', axis=1)
    
    # Create two side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Plot for NYT
    nyt_pivot.plot(kind='bar', stacked=True, ax=ax1, 
                  color=['gold', 'silver', '#CD7F32', '#1f77b4'])
    ax1.set_title('NYT: Medal Types Associated with Countries', fontsize=15)
    ax1.set_xlabel('Country', fontsize=15)
    ax1.set_ylabel('Number of Associations', fontsize=14)
    ax1.legend(['Gold', 'Silver', 'Bronze', 'General'], title='Medal Type', fontsize=14)
    ax1.tick_params(axis='x', labelsize=14)
    ax1.tick_params(axis='y', labelsize=14)
    
    # Plot for People's Daily
    pd_pivot.plot(kind='bar', stacked=True, ax=ax2,
                 color=['gold', 'silver', '#CD7F32', '#ff7f0e'])
    ax2.set_title('People\'s Daily: Medal Types Associated with Countries', fontsize=15)
    ax2.set_xlabel('Country', fontsize=15)
    ax2.set_ylabel('Number of Associations', fontsize=14)
    ax2.legend(['Gold', 'Silver', 'Bronze', 'General'], title='Medal Type', fontsize=14)
    ax2.tick_params(axis='x', labelsize=14) 
    ax2.tick_params(axis='y', labelsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'medal_country_associations.png'), dpi=300)
    plt.close()
    
    # Create additional comparison focusing on home country vs rival country
    home_rival_comparison(nyt_country_medal_assoc, pd_country_medal_assoc)

def home_rival_comparison(nyt_country_medal_assoc, pd_country_medal_assoc):
    """Create a specialized comparison of how each media source covers home country vs rival."""
    print("Generating home vs. rival country medal coverage comparison...")
    
    # For NYT: USA (home) vs China (rival)
    # For People's Daily: China (home) vs USA (rival)
    
    # Extract data
    nyt_usa = nyt_country_medal_assoc.get('usa', {'gold_medal': 0, 'silver_medal': 0, 'bronze_medal': 0, 'general_medal': 0})
    nyt_china = nyt_country_medal_assoc.get('china', {'gold_medal': 0, 'silver_medal': 0, 'bronze_medal': 0, 'general_medal': 0})
    
    pd_china = pd_country_medal_assoc.get('china', {'gold_medal': 0, 'silver_medal': 0, 'bronze_medal': 0, 'general_medal': 0})
    pd_usa = pd_country_medal_assoc.get('usa', {'gold_medal': 0, 'silver_medal': 0, 'bronze_medal': 0, 'general_medal': 0})
    
    # Calculate gold medal to total medal ratio
    def calculate_gold_ratio(medal_dict):
        total = sum(medal_dict.values())
        return medal_dict['gold_medal'] / total if total > 0 else 0
    
    nyt_usa_gold_ratio = calculate_gold_ratio(nyt_usa)
    nyt_china_gold_ratio = calculate_gold_ratio(nyt_china)
    pd_china_gold_ratio = calculate_gold_ratio(pd_china)
    pd_usa_gold_ratio = calculate_gold_ratio(pd_usa)
    
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    
    labels = ['Own Country', 'Rival Country']
    nyt_values = [nyt_usa_gold_ratio, nyt_china_gold_ratio]
    pd_values = [pd_china_gold_ratio, pd_usa_gold_ratio]
    
    x = np.arange(len(labels))
    width = 0.35
    
    plt.bar(x - width/2, nyt_values, width, label='NYT (USA)', color='#1f77b4')
    plt.bar(x + width/2, pd_values, width, label='People\'s Daily (China)', color='#ff7f0e')
    
    plt.title('Gold Medal Focus in Coverage of Own Country vs. Rival Country', fontsize=16)
    plt.ylabel('Gold Medals / Total Medal Mentions', fontsize=12)
    plt.xticks(x, labels, fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Format y-axis as percentage
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
    
    # Add value labels
    for i, v in enumerate(nyt_values):
        plt.text(i - width/2, v + 0.02, f'{v:.1%}', ha='center', fontsize=10)
    
    for i, v in enumerate(pd_values):
        plt.text(i + width/2, v + 0.02, f'{v:.1%}', ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'home_vs_rival_gold_ratio.png'), dpi=300)
    plt.close()

def generate_context_word_clouds(nyt_contexts, pd_contexts):
    """Generate word clouds from medal mention contexts."""
    print("Generating context word clouds...")
    
    # Function to create word frequency dictionary from contexts
    def get_word_freq(contexts, is_english=True):
        if is_english:
            # Preprocess English contexts
            all_words = []
            for context in contexts:
                words = [word.lower() for word in word_tokenize(context) 
                         if word.isalpha() and word.lower() not in english_stopwords 
                         and len(word) > 2]
                all_words.extend(words)
            
            # Remove medal-related terms that would dominate the word cloud
            filtered_words = [w for w in all_words if w not in ['gold', 'medal', 'medals', 'olympic', 'olympics']]
            
            return Counter(filtered_words)
        else:
            # Preprocess Chinese contexts
            all_words = []
            for context in contexts:
                words = [word for word in jieba.lcut(context) 
                         if word not in chinese_stopwords and len(word) > 1]
                all_words.extend(words)
            
            # Remove medal-related terms that would dominate the word cloud
            filtered_words = [w for w in all_words if w not in ['金牌', '银牌', '铜牌', '奖牌', '冠军', '奥运会']]
            
            return Counter(filtered_words)
    
    # Create word clouds for NYT gold medal contexts
    nyt_gold_freq = get_word_freq(nyt_contexts['gold_contexts'], is_english=True)
    if nyt_gold_freq:
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='Blues',
            contour_width=1, contour_color='steelblue'
        ).generate_from_frequencies(nyt_gold_freq)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Words Associated with Gold Medals in NYT', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'nyt_gold_context_wordcloud.png'), dpi=300)
        plt.close()
    
    # Create word clouds for NYT general medal contexts
    nyt_general_freq = get_word_freq(nyt_contexts['general_medal_contexts'], is_english=True)
    if nyt_general_freq:
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='Blues',
            contour_width=1, contour_color='steelblue'
        ).generate_from_frequencies(nyt_general_freq)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Words Associated with General Medal Mentions in NYT', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'nyt_general_context_wordcloud.png'), dpi=300)
        plt.close()
    
    # Create word clouds for People's Daily gold medal contexts
    pd_gold_freq = get_word_freq(pd_contexts['gold_contexts'], is_english=False)
    if pd_gold_freq:
        # Create a separate figure for Chinese wordcloud with proper font support
        try:
            # Try to find a suitable Chinese font
            font_path = None
            # Common Chinese font paths on Linux systems
            possible_fonts = [
                '/usr/share/fonts/truetype/arphic/uming.ttc',  # Ubuntu/Debian
                '/usr/share/fonts/wqy-microhei/wqy-microhei.ttc',  # Common on many Linux
                '/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc',
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # Noto fonts
                '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc'
            ]
            
            for font in possible_fonts:
                if os.path.exists(font):
                    font_path = font
                    break
                    
            if font_path:
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    max_words=100,
                    colormap='YlOrBr',
                    contour_width=1, contour_color='darkorange',
                    font_path=font_path
                ).generate_from_frequencies(pd_gold_freq)
            else:
                # If no Chinese font found, save frequencies to file for later processing
                with open(os.path.join(OUTPUT_PATH, 'pd_gold_context_freqs.pkl'), 'wb') as f:
                    pickle.dump(pd_gold_freq, f)
                print("Warning: No suitable Chinese font found. Word frequencies saved to file for later processing.")
                # Create basic wordcloud without proper Chinese rendering
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    max_words=100,
                    colormap='YlOrBr',
                    contour_width=1, contour_color='darkorange'
                ).generate_from_frequencies(pd_gold_freq)
        except Exception as e:
            print(f"Error generating Chinese wordcloud: {e}")
            # Save frequencies to file
            with open(os.path.join(OUTPUT_PATH, 'pd_gold_context_freqs.pkl'), 'wb') as f:
                pickle.dump(pd_gold_freq, f)
            print("Word frequencies saved to file for later processing.")
            
            # Create an empty wordcloud with message
            plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, "Chinese wordcloud could not be generated.\nFrequencies saved to pd_gold_context_freqs.pkl", 
                     ha='center', va='center', fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_PATH, 'pd_gold_context_wordcloud.png'), dpi=300)
            plt.close()
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Words Associated with Gold Medals in People\'s Daily', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'pd_gold_context_wordcloud.png'), dpi=300)
        plt.close()
    
    # Create word clouds for People's Daily general medal contexts
    pd_general_freq = get_word_freq(pd_contexts['general_medal_contexts'], is_english=False)
    if pd_general_freq:
        # Create a separate figure for Chinese wordcloud with proper font support
        try:
            # Try to find a suitable Chinese font
            font_path = None
            # Common Chinese font paths on Linux systems
            possible_fonts = [
                '/usr/share/fonts/truetype/arphic/uming.ttc',  # Ubuntu/Debian
                '/usr/share/fonts/wqy-microhei/wqy-microhei.ttc',  # Common on many Linux
                '/usr/share/fonts/wqy-zenhei/wqy-zenhei.ttc',
                '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',  # Noto fonts
                '/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc'
            ]
            
            for font in possible_fonts:
                if os.path.exists(font):
                    font_path = font
                    break
                    
            if font_path:
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    max_words=100,
                    colormap='YlOrBr',
                    contour_width=1, contour_color='darkorange',
                    font_path=font_path
                ).generate_from_frequencies(pd_general_freq)
            else:
                # If no Chinese font found, save frequencies to file for later processing
                with open(os.path.join(OUTPUT_PATH, 'pd_general_context_freqs.pkl'), 'wb') as f:
                    pickle.dump(pd_general_freq, f)
                print("Warning: No suitable Chinese font found. Word frequencies saved to file for later processing.")
                # Create basic wordcloud without proper Chinese rendering
                wordcloud = WordCloud(
                    width=800, height=400,
                    background_color='white',
                    max_words=100,
                    colormap='YlOrBr',
                    contour_width=1, contour_color='darkorange'
                ).generate_from_frequencies(pd_general_freq)
        except Exception as e:
            print(f"Error generating Chinese wordcloud: {e}")
            # Save frequencies to file
            with open(os.path.join(OUTPUT_PATH, 'pd_general_context_freqs.pkl'), 'wb') as f:
                pickle.dump(pd_general_freq, f)
            print("Word frequencies saved to file for later processing.")
            
            # Create an empty wordcloud with message
            plt.figure(figsize=(12, 6))
            plt.text(0.5, 0.5, "Chinese wordcloud could not be generated.\nFrequencies saved to pd_general_context_freqs.pkl", 
                     ha='center', va='center', fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_PATH, 'pd_general_context_wordcloud.png'), dpi=300)
            plt.close()
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Words Associated with General Medals in People\'s Daily', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_PATH, 'pd_general_context_wordcloud.png'), dpi=300)
        plt.close()

def plot_medal_mentions_olympic_vs_non_olympic(nyt_results, pd_results):
    """Compare medal mentions between Olympic and non-Olympic years."""
    print("Generating Olympic vs non-Olympic years medal comparison...")
    
    # Convert to DataFrame
    nyt_df = pd.DataFrame(nyt_results)
    pd_df = pd.DataFrame(pd_results)
    
    # Calculate average medal mentions per article for Olympic/non-Olympic years
    nyt_olympic = nyt_df[nyt_df['is_olympic_year']]
    nyt_non_olympic = nyt_df[~nyt_df['is_olympic_year']]
    
    pd_olympic = pd_df[pd_df['is_olympic_year']]
    pd_non_olympic = pd_df[~pd_df['is_olympic_year']]
    
    # Calculate medal mentions per 1000 Words for fair comparison
    def calc_medal_density(df, medal_col):
        """Calculate medal mentions per 1000 words for fair comparison.
        For NYT, uses English words; for People's Daily, uses words segmented by jieba."""
        total_mentions = df[medal_col].sum()
        total_words = df['word_count'].sum()
        return (total_mentions / total_words) * 1000 if total_words > 0 else 0
    
    # Calculate medal densities
    medal_types = ['gold_medal', 'silver_medal', 'bronze_medal', 'general_medal']
    
    nyt_olympic_densities = [calc_medal_density(nyt_olympic, medal_type) for medal_type in medal_types]
    nyt_non_olympic_densities = [calc_medal_density(nyt_non_olympic, medal_type) for medal_type in medal_types]
    
    pd_olympic_densities = [calc_medal_density(pd_olympic, medal_type) for medal_type in medal_types]
    pd_non_olympic_densities = [calc_medal_density(pd_non_olympic, medal_type) for medal_type in medal_types]
    
    # Create plot
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))
    
    # NYT Plot
    x = np.arange(len(medal_types))
    width = 0.35
    
    axs[0].bar(x - width/2, nyt_olympic_densities, width, label='Olympic Years', color='#1f77b4')
    axs[0].bar(x + width/2, nyt_non_olympic_densities, width, label='Non-Olympic Years', color='#aec7e8')
    
    axs[0].set_title('NYT: Medal Mentions Density', fontsize=14)
    axs[0].set_xlabel('Medal Type', fontsize=12)
    axs[0].set_ylabel('Mentions per 1000 Words', fontsize=12)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(['Gold', 'Silver', 'Bronze', 'General'])
    axs[0].legend()
    axs[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # People's Daily Plot
    axs[1].bar(x - width/2, pd_olympic_densities, width, label='Olympic Years', color='#ff7f0e')
    axs[1].bar(x + width/2, pd_non_olympic_densities, width, label='Non-Olympic Years', color='#ffbb78')
    
    axs[1].set_title('People\'s Daily: Medal Mentions Density', fontsize=14)
    axs[1].set_xlabel('Medal Type', fontsize=12)
    axs[1].set_ylabel('Mentions per 1000 Words', fontsize=12)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(['Gold', 'Silver', 'Bronze', 'General'])
    axs[1].legend()
    axs[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'medal_mentions_olympic_vs_non_olympic.png'), dpi=300)
    plt.close()

# -----------------------------------------------------------------------
# 6. Main Execution
# -----------------------------------------------------------------------

def main():
    """Main execution function."""
    start_time = time.time()
    
    # Print medal dictionaries for verification
    print("\nEnglish Medal Dictionary:")
    for category, words in english_medal_dict.items():
        print(f"  {category}: {len(words)} terms")
    
    print("\nChinese Medal Dictionary:")
    for category, words in chinese_medal_dict.items():
        print(f"  {category}: {len(words)} terms")
    
    # For initial development, use a smaller sample size to test the code
    # You can adjust these values or set to None to process all articles
    nyt_sample_size = None  # Set to None to process all articles
    pd_sample_size = None   # Set to None to process all articles
    
    # Run analysis for NYT
    nyt_results_path = os.path.join(OUTPUT_PATH, "nyt_medal_results.pkl")
    nyt_contexts_path = os.path.join(OUTPUT_PATH, "nyt_medal_contexts.pkl")
    
    if os.path.exists(nyt_results_path) and os.path.exists(nyt_contexts_path):
        print(f"\nLoading existing NYT results from {nyt_results_path}")
        nyt_results = pd.read_pickle(nyt_results_path)
        nyt_contexts = pd.read_pickle(nyt_contexts_path)
    else:
        nyt_results, nyt_contexts = analyze_nyt_articles_parallel(nyt_sample_size)
        with open(nyt_results_path, 'wb') as f:
            pickle.dump(nyt_results, f)
        with open(nyt_contexts_path, 'wb') as f:
            pickle.dump(nyt_contexts, f)
        print(f"Saved NYT results to {nyt_results_path}")
    
    # Run analysis for People's Daily
    pd_results_path = os.path.join(OUTPUT_PATH, "pd_medal_results.pkl")
    pd_contexts_path = os.path.join(OUTPUT_PATH, "pd_medal_contexts.pkl")
    
    if os.path.exists(pd_results_path) and os.path.exists(pd_contexts_path):
        print(f"\nLoading existing People's Daily results from {pd_results_path}")
        pd_results = pd.read_pickle(pd_results_path)
        pd_contexts = pd.read_pickle(pd_contexts_path)
    else:
        pd_results, pd_contexts = analyze_peoples_daily_articles_parallel(pd_sample_size)
        with open(pd_results_path, 'wb') as f:
            pickle.dump(pd_results, f)
        with open(pd_contexts_path, 'wb') as f:
            pickle.dump(pd_contexts, f)
        print(f"Saved People's Daily results to {pd_results_path}")
    
    # Generate visualizations
    plot_medal_type_comparison(nyt_results, pd_results)
    plot_gold_ratio_over_time(nyt_results, pd_results)
    plot_medal_mentions_by_country(nyt_results, pd_results)
    plot_medal_mentions_olympic_vs_non_olympic(nyt_results, pd_results)
    generate_context_word_clouds(nyt_contexts, pd_contexts)
    
    # Calculate and print basic summary statistics
    nyt_df = pd.DataFrame(nyt_results)
    pd_df = pd.DataFrame(pd_results)
    
    print("\nSummary Statistics:")
    
    print("\nNYT Medal Mentions:")
    print(f"Total articles analyzed: {len(nyt_df)}")
    print(f"Articles with medal mentions: {len(nyt_df[nyt_df['total_medal_mentions'] > 0])}")
    print(f"Gold medal mentions: {nyt_df['gold_medal'].sum()}")
    print(f"Silver medal mentions: {nyt_df['silver_medal'].sum()}")
    print(f"Bronze medal mentions: {nyt_df['bronze_medal'].sum()}")
    print(f"General medal mentions: {nyt_df['general_medal'].sum()}")
    print(f"Gold/Total ratio: {nyt_df['gold_medal'].sum() / nyt_df['total_medal_mentions'].sum():.2%}")
    
    print("\nPeople's Daily Medal Mentions:")
    print(f"Total articles analyzed: {len(pd_df)}")
    print(f"Articles with medal mentions: {len(pd_df[pd_df['total_medal_mentions'] > 0])}")
    print(f"Gold medal mentions: {pd_df['gold_medal'].sum()}")
    print(f"Silver medal mentions: {pd_df['silver_medal'].sum()}")
    print(f"Bronze medal mentions: {pd_df['bronze_medal'].sum()}")
    print(f"General medal mentions: {pd_df['general_medal'].sum()}")
    print(f"Gold/Total ratio: {pd_df['gold_medal'].sum() / pd_df['total_medal_mentions'].sum():.2%}")
    
    total_time = time.time() - start_time
    print(f"\nTotal analysis time: {total_time:.2f} seconds")
    print("\nAnalysis complete! Visualizations saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    # Use 'spawn' method for more reliable multiprocessing in large workloads
    mp.set_start_method('spawn', force=True)
    main()