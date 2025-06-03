# Project 3: Sentiment Analysis of Olympic Medal Coverage
# =========================================================
# This script analyzes sentiment patterns in Olympic coverage from 
# The New York Times (NYT) and People's Daily (China).
# Using multilingual BERT models to ensure cross-language comparability,
# it examines how media from both countries frame victory and defeat.
# GPU-accelerated for high-performance processing.

import os
import pandas as pd
import numpy as np
import re
import jieba
import nltk
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import pickle
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import multiprocessing as mp
import math
from sklearn.metrics import confusion_matrix, classification_report
import warnings
import threading
from scipy import stats  # Import for statistical testing
warnings.filterwarnings("ignore")

# Set paths for Linux environment
DATA_PATH = "/U_PZL2021KF0012/hx/EPF/History_and_digital/project_data/processed_data"
MEDAL_ANALYSIS_PATH = "/U_PZL2021KF0012/hx/EPF/History_and_digital/project_data/medal_analysis"
OUTPUT_PATH = "/U_PZL2021KF0012/hx/EPF/History_and_digital/project_data/sentiment_analysis"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Set NLTK data path
nltk_data_dir = "/U_PZL2021KF0012/hx/EPF/History_and_digital/nltk_data"
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

# Configure GPU settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    print(f"Memory cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")

# Maximum CPU cores for parallel processing (for data preparation)
MAX_CPUS = min(mp.cpu_count(), 128)  # Limiting to 64 cores as we'll primarily use GPU
print(f"Setting up parallel processing with {MAX_CPUS} CPU cores for data preparation")

# Define Olympic years for analysis
summer_olympic_years = [1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012]
winter_olympic_years = [1980, 1984, 1988, 1992, 1994, 1998, 2002, 2006, 2010, 2014]
olympic_years = sorted(list(set(summer_olympic_years + winter_olympic_years)))
beijing_olympics_year = 2008

# -----------------------------------------------------------------------
# 1. Load Data and Setup Models
# -----------------------------------------------------------------------

def load_datasets():
    """Load previously processed datasets and medal analysis results."""
    print("Loading datasets...")
    
    # Load standardized article datasets
    nyt_path = os.path.join(DATA_PATH, "nyt_standardized.pkl")
    pd_path = os.path.join(DATA_PATH, "people_daily_standardized.pkl")
    
    nyt_df = pd.read_pickle(nyt_path)
    pd_df = pd.read_pickle(pd_path)
    
    # Mark Olympic years
    nyt_df['is_olympic_year'] = nyt_df['year'].isin(olympic_years)
    pd_df['is_olympic_year'] = pd_df['year'].isin(olympic_years)
    
    # Load medal analysis results
    nyt_medal_results_path = os.path.join(MEDAL_ANALYSIS_PATH, "nyt_medal_results.pkl")
    pd_medal_results_path = os.path.join(MEDAL_ANALYSIS_PATH, "pd_medal_results.pkl")
    
    nyt_medal_results = pd.read_pickle(nyt_medal_results_path)
    pd_medal_results = pd.read_pickle(pd_medal_results_path)
    
    # Load medal contexts
    nyt_medal_contexts_path = os.path.join(MEDAL_ANALYSIS_PATH, "nyt_medal_contexts.pkl")
    pd_medal_contexts_path = os.path.join(MEDAL_ANALYSIS_PATH, "pd_medal_contexts.pkl")
    
    nyt_medal_contexts = pd.read_pickle(nyt_medal_contexts_path)
    pd_medal_contexts = pd.read_pickle(pd_medal_contexts_path)
    
    print(f"NYT dataset: {len(nyt_df)} articles")
    print(f"People's Daily dataset: {len(pd_df)} articles")
    
    return nyt_df, pd_df, nyt_medal_results, pd_medal_results, nyt_medal_contexts, pd_medal_contexts

def setup_sentiment_model(gpu_id=0):
    """Initialize the multilingual sentiment analysis model on a specific GPU."""
    print(f"Setting up multilingual sentiment analysis model on GPU {gpu_id}...")
    
    # Specify the device to use
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model_path = "/U_PZL2021KF0012/hx/.cache/huggingface/hub/models--cardiffnlp--twitter-xlm-roberta-base-sentiment/snapshots/f2f1202b1bdeb07342385c3f807f9c07cd8f5cf8/"
    
    from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification
    
    # Explicitly use slow tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_path)
    model = XLMRobertaForSequenceClassification.from_pretrained(model_path)
    
    # Move model to specified GPU
    model.to(device)
    
    # Create sentiment analysis pipeline
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model=model, 
        tokenizer=tokenizer, 
        device=gpu_id if torch.cuda.is_available() else -1
    )
    
    print(f"Sentiment model setup complete on GPU {gpu_id}")
    return device, tokenizer, model, sentiment_analyzer

# -----------------------------------------------------------------------
# 2. Context Extraction Functions
# -----------------------------------------------------------------------

# Lists of country terms for filtering
english_countries = {
    'usa': ['united states', 'usa', 'u.s.', 'u.s.a.', 'american', 'americans', 'america', 'team usa'],
    'china': ['china', 'chinese', 'prc', 'people\'s republic of china', 'team china'],
}

chinese_countries = {
    'usa': ['美国', '美国队', '美国人', '美利坚', '美'],
    'china': ['中国', '中国队', '国家队', '中华', '国人', '我国', '我们', '中'],
}

# Medal terms
english_medal_terms = ['gold-winning', 'bronze medals', 'bronze medallist', 'olympic championship', 'medalist', 'silver medal', 'third-place', 'gold medallists', '2nd place', 'bronze-winning', 'medals', 'total medals', 'gold medals', 'bronze-medal', 'gold medalist', '3rd place', 'medalists', 'second-place', 'third place', 'bronze medalists', 'medal haul', 'runner-up', 'silver medallists', 'olympic silver', 'olympic title', 'bronze medalist', 'medal table', 'overall medals', 'first place', 'gold winner', 'medal count', 'olympic gold', 'medallists', 'silver medalist', 'medal totals', 'gold medal', 'first-place', 'gold-medal', 'bronze medal', 'bronze winner', 'medal race', 'gold medalists', 'runner up', 'olympic bronze', 'winning medal', 'gold medallist', 'silver-winning', 'medal standings', 'silver medallist', 'medal tally', 'silver medals', 'medal winners', 'medallist', 'second place', '1st place', 'olympic champion', 'silver winner', 'medal', 'silver medalists', 'bronze medallists', 'silver-medal', 'podium', 'medal winner', 'top three']

chinese_medal_terms = ['奖牌总榜', '摘得铜牌', '摘得银牌', '金银铜', '夺得铜牌', '屈居亚军', '领奖台', '夺冠', '奖牌数', '第三位', '获得金牌', '颁奖台', '金银铜牌', '亚军', '铜牌', '金牌得主', '第2名', '获得冠军', '登顶', '奖牌总数', '亚军获得者', '头名', '金牌', '铜牌获得者', '拿下铜牌', '奖牌统计', '夺得金牌', '总奖牌数', '冠军', '获得亚军', '获得银牌', '奖牌获得者', '奥运冠军', '季军', '银牌', '第二位', '总奖牌', '铜牌得主', '拿下银牌', '夺得银牌', '第二名', '奖牌积分', '夺魁', '银牌得主', '第3名', '奖牌榜排名', '获得季军', '获胜', '第一位', '第三名', '奖牌成绩', '摘得金牌', '获得铜牌', '第一名', '奖牌得主', '第1名', '排行榜', '实现金牌零的突破', '奖牌', '奖牌榜', '前三名', '拿下金牌']

english_success_terms = [
    # Basic victory vocabulary
    'win', 'won', 'winner', 'winning', 'victory', 'victorious', 'triumph', 'triumphant',
    
    # Medal and achievement related
    'gold', 'champion', 'championship', 'first', 'title', 'crown', 'podium',
    
    # Performance related positive words
    'record', 'personal best', 'world record', 'olympic record', 'breakthrough',
    'dominate', 'dominated', 'commanding', 'impressive', 'outstanding', 'stellar',
    
    # Breakthrough achievements
    'upset', 'surpass', 'exceed', 'overcome', 'outperform', 'beat', 'defeated',
    
    # Emotion and achievement expressions
    'glory', 'proud', 'pride', 'honor', 'jubilant', 'celebration', 'celebrate',
    'success', 'successful', 'accomplished', 'achievement', 'historic', 'legendary'
]

chinese_success_terms = [
    # Basic victory vocabulary
    '夺冠', '胜利', '获胜', '战胜', '赢', '赢得', '成功', '成功夺得',
    
    # Medal and achievement related
    '冠军', '金牌', '第一', '第一名', '称王', '称霸', '登顶', '问鼎', '摘金',
    
    # Performance related positive words
    '纪录', '打破纪录', '世界纪录', '奥运纪录', '突破', '佳绩', '好成绩',
    '统治', '主导', '出色', '优异', '卓越', '精彩', '完美', '霸气',
    
    # Breakthrough achievements
    '黑马', '爆冷', '超越', '击败', '战胜', '力克', '压倒', '胜过',
    
    # Emotion and achievement expressions
    '荣耀', '光荣', '自豪', '骄傲', '荣誉', '欢呼', '庆祝', '壮举',
    '成就', '创造历史', '传奇', '辉煌', '伟大', '圆梦', '实现梦想'
]


english_failure_terms = [
    # Basic failure vocabulary
    'lose', 'lost', 'loser', 'losing', 'defeat', 'defeated', 'fail', 'failed', 'failure',
    
    # Poor performance related
    'missed', 'miss', 'disappointing', 'disappointment', 'setback', 'upset',
    'struggle', 'struggled', 'poor', 'lackluster', 'underwhelming',
    
    # Unfulfilled expectations related
    'fall short', 'short of', 'below expectations', 'unable to', 'couldn\'t',
    'eliminated', 'knocked out', 'out of contention',
    
    # Emotional expressions
    'heartbreak', 'heartbreaking', 'devastating', 'crushed', 'painful',
    'disappointment', 'bitter', 'frustration', 'frustrated', 'dejected',
    
    # Ranking and results related
    'last place', 'behind', 'trailed', 'faltered', 'stumbled', 'collapse'
]


chinese_failure_terms = [
    # Basic failure vocabulary
    '失败', '输', '输掉', '落败', '败北', '失利', '负', '惜败',
    
    # Poor performance related
    '遗憾', '未能', '无缘', '差强人意', '不尽如人意', '欠佳', '不佳',
    '挣扎', '苦苦挣扎', '低迷', '不振', '疲软', '乏力',
    
    # Unfulfilled expectations related
    '未达预期', '未达目标', '不如预期', '不敌', '出局', '被淘汰',
    '止步', '功亏一篑', '憾别', '遗憾出局',
    
    # Emotional expressions
    '痛心', '心碎', '遗憾', '沮丧', '失望', '苦涩', '痛苦',
    '挫折', '受挫', '懊恼', '黯然', '低落',
    
    # Ranking and results related
    '垫底', '落后', '滑落', '跌出', '不敌', '不敌对手', '败给'
]

def extract_contexts_from_articles(df, is_english=True, context_window=3):
    """Extract relevant contexts from articles for sentiment analysis."""
    print(f"Extracting contexts from {'English' if is_english else 'Chinese'} articles...")
    
    countries = english_countries if is_english else chinese_countries
    medal_terms = english_medal_terms if is_english else chinese_medal_terms
    
    # Initialize containers for different context types
    contexts = {
        'medal_contexts': [],
        'usa_contexts': [],
        'china_contexts': [],
        'usa_medal_contexts': [],
        'china_medal_contexts': [],
        'success_contexts': [],
        'failure_contexts': [],
    }
    
    for _, article in tqdm(df.iterrows(), total=len(df), desc="Extracting contexts"):
        content = article['content']
        if not isinstance(content, str) or not content.strip():
            continue
            
        # Prepare metadata for each context
        metadata = {
            'article_id': article['original_id'],
            'year': article['year'],
            'month': article['month'],
            'is_olympic_year': article['is_olympic_year'],
        }
        
        # Segment text into sentences
        if is_english:
            sentences = sent_tokenize(content)
        else:
            # For Chinese, use punctuation to split into sentences
            sentences = re.split(r'[。！？；]', content)
        
        # Process each sentence
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:  # Skip very short sentences
                continue
                
            # Check for medal mentions
            has_medal = any(term in sentence.lower() for term in medal_terms)
            
            # Check for country mentions
            has_usa = any(term in sentence.lower() for term in countries['usa'])
            has_china = any(term in sentence.lower() for term in countries['china'])
            
            # Store contexts based on content
            if has_medal:
                contexts['medal_contexts'].append({**metadata, 'text': sentence, 'context_type': 'medal'})
            
            if has_usa:
                contexts['usa_contexts'].append({**metadata, 'text': sentence, 'context_type': 'usa'})
            
            if has_china:
                contexts['china_contexts'].append({**metadata, 'text': sentence, 'context_type': 'china'})
            
            if has_medal and has_usa:
                contexts['usa_medal_contexts'].append({**metadata, 'text': sentence, 'context_type': 'usa_medal'})
            
            if has_medal and has_china:
                contexts['china_medal_contexts'].append({**metadata, 'text': sentence, 'context_type': 'china_medal'})
            
            # Identify success and failure contexts (simplified heuristic)
            success_terms = english_success_terms if is_english else chinese_success_terms
            failure_terms = english_failure_terms if is_english else chinese_failure_terms 
            
            is_success = has_medal and any(term in sentence.lower() for term in success_terms)
            is_failure = any(term in sentence.lower() for term in failure_terms)
            
            if is_success:
                contexts['success_contexts'].append({**metadata, 'text': sentence, 'context_type': 'success'})
            if is_failure:
                contexts['failure_contexts'].append({**metadata, 'text': sentence, 'context_type': 'failure'})
    
    # Convert lists to DataFrames
    dataframes = {}
    for key, context_list in contexts.items():
        if context_list:  # Only create DataFrame if there are contexts
            dataframes[key] = pd.DataFrame(context_list)
            print(f"  - {key}: {len(context_list)} contexts extracted")
    
    return dataframes

def extract_medal_contexts(medal_contexts, is_english=True):
    """Process medal contexts from previous analysis."""
    gold_contexts = medal_contexts.get('gold_contexts', [])
    general_contexts = medal_contexts.get('general_medal_contexts', [])
    
    # Process gold medal contexts
    gold_df = pd.DataFrame({
        'text': gold_contexts,
        'context_type': 'gold_medal'
    })
    
    # Process general medal contexts
    general_df = pd.DataFrame({
        'text': general_contexts,
        'context_type': 'general_medal'
    })
    
    # Combine
    if not gold_df.empty and not general_df.empty:
        combined_df = pd.concat([gold_df, general_df], ignore_index=True)
        print(f"Extracted {len(combined_df)} medal contexts from previous analysis")
        return combined_df
    elif not gold_df.empty:
        print(f"Extracted {len(gold_df)} gold medal contextfs from previous analysis")
        return gold_df
    elif not general_df.empty:
        print(f"Extracted {len(general_df)} general medal contexts from previous analysis")
        return general_df
    else:
        print("No medal contexts found from previous analysis")
        return pd.DataFrame()

# -----------------------------------------------------------------------
# 3. Sentiment Analysis Functions
# -----------------------------------------------------------------------

class SentimentDataset(Dataset):
    """PyTorch Dataset for sentiment analysis."""
    def __init__(self, texts, tokenizer, max_length=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Remove batch dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        return item

def analyze_sentiment_batch(texts, tokenizer, model, device, batch_size=128, num_workers=8):
    """Analyze sentiment of texts in batches using PyTorch DataLoader with optimized settings."""
    # Create dataset and dataloader with optimized settings
    dataset = SentimentDataset(texts, tokenizer)
    
    # Use larger batch_size and multiple workers to speed up data loading
    # pin_memory=True can accelerate CPU to GPU data transfer
    # prefetch_factor increases the number of batches to prefetch
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,  # Use multiple worker processes to load data
        pin_memory=True,          # Speed up CPU to GPU data transfer
        prefetch_factor=2,        # Number of batches to prefetch per worker
        persistent_workers=True   # Keep worker processes alive to avoid repetitive creation/destruction
    )
    
    # Predictions container
    all_predictions = []
    all_scores = []
    
    # Process batches
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing sentiment in batches"):
            # Move batch to device
            inputs = {key: val.to(device) for key, val in batch.items()}
            
            # Model inference
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Get predictions and scores
            predictions = torch.argmax(logits, dim=1)
            scores = torch.softmax(logits, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_scores.extend(scores.cpu().numpy())
    
    # Map predictions to labels
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    sentiment_labels = [label_map[pred] for pred in all_predictions]
    
    # Extract confidence scores for each class
    negative_scores = [score[0] for score in all_scores]
    neutral_scores = [score[1] for score in all_scores]
    positive_scores = [score[2] for score in all_scores]
    
    # Calculate compound sentiment score (-1 to 1 scale)
    compound_scores = [pos - neg for pos, neg in zip(positive_scores, negative_scores)]
    
    results = {
        'sentiment': sentiment_labels,
        'negative_score': negative_scores,
        'neutral_score': neutral_scores,
        'positive_score': positive_scores,
        'compound_score': compound_scores
    }
    
    return results


def process_context_sentiments(contexts_df, tokenizer, model, device, batch_size=256, num_workers=None):
    """Process sentiments for a context DataFrame."""
    if contexts_df.empty:
        return pd.DataFrame()
    
    # Use MAX_CPUS as default worker count
    if num_workers is None:
        num_workers = min(MAX_CPUS // 2, 8)  # Allocate a reasonable number of workers per GPU
    
    # Extract texts for sentiment analysis
    texts = contexts_df['text'].tolist()
    
    # Analyze sentiment with optimized parameters
    sentiment_results = analyze_sentiment_batch(
        texts, 
        tokenizer, 
        model, 
        device, 
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Add sentiment results to DataFrame
    for key, values in sentiment_results.items():
        contexts_df[key] = values
    
    return contexts_df

def process_nyt_sentiments(nyt_all_contexts, tokenizer_0, model_0, device_0, output_path):
    """Process NYT sentiments and save results"""
    print("Processing NYT sentiments on GPU 0...")
    # Allocate half of the CPU cores for each GPU
    num_workers = MAX_CPUS // 2
    print(f"Using {num_workers} workers for NYT processing")
    
    # For large A800 GPU, use a larger batch size
    nyt_sentiments = process_context_sentiments(
        nyt_all_contexts, 
        tokenizer_0, 
        model_0, 
        device_0, 
        batch_size=512,  # Adjust based on data size and GPU memory
        num_workers=num_workers
    )
    nyt_sentiments_path = os.path.join(output_path, "nyt_sentiments.pkl")
    nyt_sentiments.to_pickle(nyt_sentiments_path)
    print("NYT sentiment analysis completed and saved")
    return nyt_sentiments

def process_pd_sentiments(pd_all_contexts, tokenizer_1, model_1, device_1, output_path):
    """Process People's Daily sentiments and save results"""
    print("Processing People's Daily sentiments on GPU 1...")
    # Allocate half of the CPU cores for each GPU
    num_workers = MAX_CPUS // 2
    print(f"Using {num_workers} workers for PD processing")
    
    pd_sentiments = process_context_sentiments(
        pd_all_contexts, 
        tokenizer_1, 
        model_1, 
        device_1,
        batch_size=512,  # Adjust based on data size and GPU memory
        num_workers=num_workers
    )
    pd_sentiments_path = os.path.join(output_path, "pd_sentiments.pkl")
    pd_sentiments.to_pickle(pd_sentiments_path)
    print("People's Daily sentiment analysis completed and saved")
    return pd_sentiments

# -----------------------------------------------------------------------
# 4. Analysis and Visualization Functions
# -----------------------------------------------------------------------

def plot_sentiment_distribution_separate(nyt_sentiments, pd_sentiments, context_type):
    """Plot separate sentiment distributions for NYT and People's Daily."""
    print(f"Plotting separate sentiment distributions for {context_type} contexts...")
    
    # Create two separate figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Prepare data for NYT
    nyt_counts = nyt_sentiments['sentiment'].value_counts().reindex(['positive', 'neutral', 'negative']).fillna(0)
    nyt_percentages = nyt_counts / nyt_counts.sum() * 100
    
    # Prepare data for People's Daily
    pd_counts = pd_sentiments['sentiment'].value_counts().reindex(['positive', 'neutral', 'negative']).fillna(0)
    pd_percentages = pd_counts / pd_counts.sum() * 100
    
    # Plot for NYT
    x = np.arange(len(nyt_percentages.index))
    ax1.bar(x, nyt_percentages, color='#1f77b4')
    ax1.set_title(f'NYT: Sentiment Distribution in {context_type.replace("_", " ").title()} Contexts', fontsize=14)
    ax1.set_ylabel('Percentage', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Positive', 'Neutral', 'Negative'])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(nyt_percentages):
        ax1.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    # Plot for People's Daily
    ax2.bar(x, pd_percentages, color='#ff7f0e')
    ax2.set_title(f'People\'s Daily: Sentiment Distribution in {context_type.replace("_", " ").title()} Contexts', fontsize=14)
    ax2.set_ylabel('Percentage', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Positive', 'Neutral', 'Negative'])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(pd_percentages):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, f'{context_type}_sentiment_distribution.png'), dpi=300)
    plt.close()
    
    return nyt_percentages, pd_percentages

def plot_sentiment_time_series(nyt_sentiments, pd_sentiments, context_type):
    """Plot sentiment trends over time for both publications."""
    print(f"Plotting sentiment time series for {context_type} contexts...")
    
    # Prepare data
    nyt_yearly = nyt_sentiments.groupby('year')['compound_score'].mean().reset_index()
    pd_yearly = pd_sentiments.groupby('year')['compound_score'].mean().reset_index()
    
    # Create plot
    plt.figure(figsize=(14, 7))
    
    plt.plot(nyt_yearly['year'], nyt_yearly['compound_score'], 
             marker='o', linestyle='-', color='#1f77b4', label='NYT')
    plt.plot(pd_yearly['year'], pd_yearly['compound_score'], 
             marker='s', linestyle='-', color='#ff7f0e', label='People\'s Daily')
    
    # Add vertical lines for Olympic years
    for year in olympic_years:
        plt.axvline(x=year, color='gray', linestyle='--', alpha=0.5)
    
    # Add vertical line for 2008 Beijing Olympics
    plt.axvline(x=2008, color='red', linestyle='-', alpha=0.7)
    plt.text(2008, plt.ylim()[1]*0.95, 'Beijing\nOlympics', 
             ha='center', va='top', color='red', fontweight='bold')
    
    # Add labels and formatting
    plt.title(f'Sentiment Trends in {context_type.replace("_", " ").title()} Contexts (1980-2015)', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average Sentiment Score\n(Negative to Positive)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set y-axis to show full range from negative to positive
    plt.ylim(-1, 1)
    
    # Add horizontal line at y=0 (neutral sentiment)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, f'{context_type}_sentiment_time_series.png'), dpi=300)
    plt.close()

def plot_home_vs_rival_sentiment_separate(nyt_sentiments, pd_sentiments):
    """Plot separate charts for NYT and PD comparing home vs. rival country coverage."""
    print("Plotting separate home vs. rival country sentiment comparisons...")
    
    # Create two separate figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract sentiment data for USA and China contexts in NYT
    nyt_usa = nyt_sentiments[nyt_sentiments['context_type'] == 'usa_medal']['compound_score'].mean()
    nyt_china = nyt_sentiments[nyt_sentiments['context_type'] == 'china_medal']['compound_score'].mean()
    
    # Extract sentiment data for USA and China contexts in People's Daily
    pd_usa = pd_sentiments[pd_sentiments['context_type'] == 'usa_medal']['compound_score'].mean()
    pd_china = pd_sentiments[pd_sentiments['context_type'] == 'china_medal']['compound_score'].mean()
    
    # For NYT: USA is home, China is rival
    nyt_labels = ['Own Country\n(USA)', 'Rival Country\n(China)']
    nyt_values = [nyt_usa, nyt_china]
    
    # For People's Daily: China is home, USA is rival
    pd_labels = ['Own Country\n(China)', 'Rival Country\n(USA)']
    pd_values = [pd_china, pd_usa]
    
    # Plot for NYT
    x = np.arange(len(nyt_labels))
    ax1.bar(x, nyt_values, color='#1f77b4')
    ax1.set_title('NYT: Sentiment When Covering\nOwn Country vs. Rival Country', fontsize=14)
    ax1.set_ylabel('Average Sentiment Score\n(Negative to Positive)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(nyt_labels, fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(-1, 1)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels and t-test result for NYT
    for i, v in enumerate(nyt_values):
        ax1.text(i, v + 0.05 if v >= 0 else v - 0.1, f'{v:.2f}', ha='center')
    
    # Perform t-test for NYT home vs. rival
    nyt_home_samples = nyt_sentiments[nyt_sentiments['context_type'] == 'usa_medal']['compound_score']
    nyt_rival_samples = nyt_sentiments[nyt_sentiments['context_type'] == 'china_medal']['compound_score']
    t_stat, p_value = stats.ttest_ind(nyt_home_samples, nyt_rival_samples, equal_var=False)
    significance = "p < 0.05*" if p_value < 0.05 else "n.s."
    ax1.text(0.5, 0.95, f'T-test: {significance}\np={p_value:.4f}', 
             transform=ax1.transAxes, ha='center', va='top', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Plot for People's Daily
    ax2.bar(x, pd_values, color='#ff7f0e')
    ax2.set_title('People\'s Daily: Sentiment When Covering\nOwn Country vs. Rival Country', fontsize=14)
    ax2.set_ylabel('Average Sentiment Score\n(Negative to Positive)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pd_labels, fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(-1, 1)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels and t-test result for People's Daily
    for i, v in enumerate(pd_values):
        ax2.text(i, v + 0.05 if v >= 0 else v - 0.1, f'{v:.2f}', ha='center')
    
    # Perform t-test for PD home vs. rival
    pd_home_samples = pd_sentiments[pd_sentiments['context_type'] == 'china_medal']['compound_score']
    pd_rival_samples = pd_sentiments[pd_sentiments['context_type'] == 'usa_medal']['compound_score']
    t_stat, p_value = stats.ttest_ind(pd_home_samples, pd_rival_samples, equal_var=False)
    significance = "p < 0.05*" if p_value < 0.05 else "n.s."
    ax2.text(0.5, 0.95, f'T-test: {significance}\np={p_value:.4f}', 
             transform=ax2.transAxes, ha='center', va='top', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'home_vs_rival_sentiment_separate.png'), dpi=300)
    plt.close()
    
    return {
        'nyt': {'own': nyt_usa, 'rival': nyt_china, 'p_value': p_value},
        'pd': {'own': pd_china, 'rival': pd_usa, 'p_value': p_value}
    }

def plot_success_vs_failure_sentiment_separate(nyt_sentiments, pd_sentiments):
    """Plot separate success vs. failure sentiment comparisons for NYT and People's Daily."""
    print("Plotting separate success vs. failure sentiment comparisons...")
    
    # Create two separate figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Extract sentiment data for success contexts
    nyt_success = nyt_sentiments[nyt_sentiments['context_type'] == 'success']['compound_score'].mean()
    pd_success = pd_sentiments[pd_sentiments['context_type'] == 'success']['compound_score'].mean()
    
    # Extract sentiment data for failure contexts
    nyt_failure = nyt_sentiments[nyt_sentiments['context_type'] == 'failure']['compound_score'].mean()
    pd_failure = pd_sentiments[pd_sentiments['context_type'] == 'failure']['compound_score'].mean()
    
    # Calculate sentiment gap between success and failure
    nyt_gap = nyt_success - nyt_failure
    pd_gap = pd_success - pd_failure
    
    # For NYT
    nyt_labels = ['Success', 'Failure']
    nyt_values = [nyt_success, nyt_failure]
    
    # For People's Daily
    pd_labels = ['Success', 'Failure']
    pd_values = [pd_success, pd_failure]
    
    # Create plot for NYT
    x = np.arange(len(nyt_labels))
    ax1.bar(x, nyt_values, color='#1f77b4')
    ax1.set_title('NYT: Sentiment in Success vs. Failure Contexts', fontsize=14)
    ax1.set_ylabel('Average Sentiment Score\n(Negative to Positive)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(nyt_labels, fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(-1, 1)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(nyt_values):
        ax1.text(i, v + 0.05 if v >= 0 else v - 0.1, f'{v:.2f}', ha='center')
    
    # Perform t-test for NYT success vs. failure
    nyt_success_samples = nyt_sentiments[nyt_sentiments['context_type'] == 'success']['compound_score']
    nyt_failure_samples = nyt_sentiments[nyt_sentiments['context_type'] == 'failure']['compound_score']
    t_stat, p_value = stats.ttest_ind(nyt_success_samples, nyt_failure_samples, equal_var=False)
    significance = "p < 0.05*" if p_value < 0.05 else "n.s."
    ax1.text(0.5, 0.95, f'Gap: {nyt_gap:.2f}\nT-test: {significance}\np={p_value:.4f}', 
             transform=ax1.transAxes, ha='center', va='top', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Create plot for People's Daily
    ax2.bar(x, pd_values, color='#ff7f0e')
    ax2.set_title('People\'s Daily: Sentiment in Success vs. Failure Contexts', fontsize=14)
    ax2.set_ylabel('Average Sentiment Score\n(Negative to Positive)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(pd_labels, fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(-1, 1)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(pd_values):
        ax2.text(i, v + 0.05 if v >= 0 else v - 0.1, f'{v:.2f}', ha='center')
    
    # Perform t-test for PD success vs. failure
    pd_success_samples = pd_sentiments[pd_sentiments['context_type'] == 'success']['compound_score']
    pd_failure_samples = pd_sentiments[pd_sentiments['context_type'] == 'failure']['compound_score']
    t_stat, p_value = stats.ttest_ind(pd_success_samples, pd_failure_samples, equal_var=False)
    significance = "p < 0.05*" if p_value < 0.05 else "n.s."
    ax2.text(0.5, 0.95, f'Gap: {pd_gap:.2f}\nT-test: {significance}\np={p_value:.4f}', 
             transform=ax2.transAxes, ha='center', va='top', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'success_vs_failure_sentiment_separate.png'), dpi=300)
    plt.close()
    
    # Also create the sentiment gap comparison chart
    fig, ax = plt.subplots(figsize=(8, 6))
    gaps = [nyt_gap, pd_gap]
    labels = ['NYT', 'People\'s Daily']
    
    ax.bar(labels, gaps, color=['#1f77b4', '#ff7f0e'])
    ax.set_title('Sentiment Gap (Success - Failure)', fontsize=16)
    ax.set_ylabel('Sentiment Difference', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(gaps):
        ax.text(i, v + 0.05 if v >= 0 else v - 0.1, f'{v:.2f}', ha='center')
    
    # Perform t-test to compare the gaps
    # This requires bootstrapping or another approach to compare the differences
    # For simplicity, we'll just note the values
    ax.text(0.5, 0.95, f'Gap difference: {abs(nyt_gap - pd_gap):.2f}', 
            transform=ax.transAxes, ha='center', va='top', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'sentiment_gap_comparison.png'), dpi=300)
    plt.close()
    
    return {
        'nyt': {'success': nyt_success, 'failure': nyt_failure, 'gap': nyt_gap, 'p_value': p_value},
        'pd': {'success': pd_success, 'failure': pd_failure, 'gap': pd_gap, 'p_value': p_value}
    }

def plot_beijing_olympics_impact_separate(nyt_sentiments, pd_sentiments):
    """Plot separate charts for NYT and PD showing sentiment changes around Beijing Olympics."""
    print("Plotting separate analyses of Beijing Olympics impact on sentiment...")
    
    # Define periods for analysis
    pre_beijing = [2004, 2005, 2006, 2007]  # Pre-Beijing Olympics
    beijing_year = [2008]                   # Beijing Olympics year
    post_beijing = [2009, 2010, 2011, 2012] # Post-Beijing Olympics
    
    # Filter results for these periods
    nyt_pre = nyt_sentiments[nyt_sentiments['year'].isin(pre_beijing)]
    nyt_during = nyt_sentiments[nyt_sentiments['year'].isin(beijing_year)]
    nyt_post = nyt_sentiments[nyt_sentiments['year'].isin(post_beijing)]
    
    pd_pre = pd_sentiments[pd_sentiments['year'].isin(pre_beijing)]
    pd_during = pd_sentiments[pd_sentiments['year'].isin(beijing_year)]
    pd_post = pd_sentiments[pd_sentiments['year'].isin(post_beijing)]
    
    # Calculate mean sentiment for each period
    nyt_values = [
        nyt_pre['compound_score'].mean(),
        nyt_during['compound_score'].mean(),
        nyt_post['compound_score'].mean()
    ]
    
    pd_values = [
        pd_pre['compound_score'].mean(),
        pd_during['compound_score'].mean(),
        pd_post['compound_score'].mean()
    ]
    
    # Create two separate figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Labels for x-axis
    periods = ['2004-2007\n(Pre-Beijing)', '2008\n(Beijing Olympics)', '2009-2012\n(Post-Beijing)']
    x = np.arange(len(periods))
    
    # Plot for NYT
    ax1.bar(x, nyt_values, color='#1f77b4')
    ax1.set_title('NYT: Impact of 2008 Beijing Olympics on Sentiment', fontsize=14)
    ax1.set_ylabel('Average Sentiment Score\n(Negative to Positive)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(periods, fontsize=10)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(-1, 1)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(nyt_values):
        ax1.text(i, v + 0.05 if v >= 0 else v - 0.1, f'{v:.2f}', ha='center')
    
    # Perform ANOVA test for NYT periods
    nyt_samples = [
        nyt_pre['compound_score'].values,
        nyt_during['compound_score'].values,
        nyt_post['compound_score'].values
    ]
    f_stat, p_value = stats.f_oneway(*nyt_samples)
    significance = "p < 0.05*" if p_value < 0.05 else "n.s."
    ax1.text(0.5, 0.95, f'ANOVA: {significance}\np={p_value:.4f}', 
             transform=ax1.transAxes, ha='center', va='top', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Plot for People's Daily
    ax2.bar(x, pd_values, color='#ff7f0e')
    ax2.set_title('People\'s Daily: Impact of 2008 Beijing Olympics on Sentiment', fontsize=14)
    ax2.set_ylabel('Average Sentiment Score\n(Negative to Positive)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(periods, fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(-1, 1)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(pd_values):
        ax2.text(i, v + 0.05 if v >= 0 else v - 0.1, f'{v:.2f}', ha='center')
    
    # Perform ANOVA test for PD periods
    pd_samples = [
        pd_pre['compound_score'].values,
        pd_during['compound_score'].values,
        pd_post['compound_score'].values
    ]
    f_stat, p_value = stats.f_oneway(*pd_samples)
    significance = "p < 0.05*" if p_value < 0.05 else "n.s."
    ax2.text(0.5, 0.95, f'ANOVA: {significance}\np={p_value:.4f}', 
             transform=ax2.transAxes, ha='center', va='top', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'beijing_olympics_sentiment_impact_separate.png'), dpi=300)
    plt.close()
    
    # Perform t-tests between periods
    results = {
        'nyt': {
            'pre_during': stats.ttest_ind(nyt_pre['compound_score'], nyt_during['compound_score'], equal_var=False),
            'during_post': stats.ttest_ind(nyt_during['compound_score'], nyt_post['compound_score'], equal_var=False),
            'pre_post': stats.ttest_ind(nyt_pre['compound_score'], nyt_post['compound_score'], equal_var=False),
            'values': nyt_values,
            'anova': (f_stat, p_value)
        },
        'pd': {
            'pre_during': stats.ttest_ind(pd_pre['compound_score'], pd_during['compound_score'], equal_var=False),
            'during_post': stats.ttest_ind(pd_during['compound_score'], pd_post['compound_score'], equal_var=False),
            'pre_post': stats.ttest_ind(pd_pre['compound_score'], pd_post['compound_score'], equal_var=False),
            'values': pd_values,
            'anova': (f_stat, p_value)
        }
    }
    
    return results

def analyze_gold_vs_general_sentiment_separate(nyt_sentiments, pd_sentiments):
    """Create separate charts comparing gold medal vs. general medal sentiment for each publication."""
    print("Analyzing gold vs. general medal sentiment separately...")
    
    # Filter for gold and general medal contexts
    nyt_gold = nyt_sentiments[nyt_sentiments['context_type'] == 'gold_medal']
    nyt_general = nyt_sentiments[nyt_sentiments['context_type'] == 'general_medal']
    
    pd_gold = pd_sentiments[pd_sentiments['context_type'] == 'gold_medal']
    pd_general = pd_sentiments[pd_sentiments['context_type'] == 'general_medal']
    
    # Calculate mean sentiment
    nyt_gold_mean = nyt_gold['compound_score'].mean()
    nyt_general_mean = nyt_general['compound_score'].mean()
    
    pd_gold_mean = pd_gold['compound_score'].mean()
    pd_general_mean = pd_general['compound_score'].mean()
    
    # Create two separate figures
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Labels for x-axis
    labels = ['Gold Medal', 'General Medal']
    x = np.arange(len(labels))
    
    # Plot for NYT
    nyt_values = [nyt_gold_mean, nyt_general_mean]
    ax1.bar(x, nyt_values, color='#1f77b4')
    ax1.set_title('NYT: Gold Medal vs. General Medal Mentions', fontsize=14)
    ax1.set_ylabel('Average Sentiment Score\n(Negative to Positive)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(-1, 1)
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(nyt_values):
        ax1.text(i, v + 0.05 if v >= 0 else v - 0.1, f'{v:.2f}', ha='center')
    
    # Perform t-test for NYT gold vs. general
    t_stat, p_value = stats.ttest_ind(nyt_gold['compound_score'], nyt_general['compound_score'], equal_var=False)
    significance = "p < 0.05*" if p_value < 0.05 else "n.s."
    ax1.text(0.5, 0.95, f'T-test: {significance}\np={p_value:.4f}', 
             transform=ax1.transAxes, ha='center', va='top', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    # Plot for People's Daily
    pd_values = [pd_gold_mean, pd_general_mean]
    ax2.bar(x, pd_values, color='#ff7f0e')
    ax2.set_title('People\'s Daily: Gold Medal vs. General Medal Mentions', fontsize=14)
    ax2.set_ylabel('Average Sentiment Score\n(Negative to Positive)', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim(-1, 1)
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(pd_values):
        ax2.text(i, v + 0.05 if v >= 0 else v - 0.1, f'{v:.2f}', ha='center')
    
    # Perform t-test for PD gold vs. general
    t_stat, p_value = stats.ttest_ind(pd_gold['compound_score'], pd_general['compound_score'], equal_var=False)
    significance = "p < 0.05*" if p_value < 0.05 else "n.s."
    ax2.text(0.5, 0.95, f'T-test: {significance}\np={p_value:.4f}', 
             transform=ax2.transAxes, ha='center', va='top', fontsize=10, 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'gold_vs_general_sentiment_separate.png'), dpi=300)
    plt.close()
    
    return {
        'nyt': {'gold': nyt_gold_mean, 'general': nyt_general_mean, 'p_value': p_value},
        'pd': {'gold': pd_gold_mean, 'general': pd_general_mean, 'p_value': p_value}
    }

def generate_sentiment_summary_statistics(nyt_sentiments, pd_sentiments, statistical_tests=None):
    """Generate and save summary statistics for sentiment analysis, including statistical test results."""
    print("Generating sentiment summary statistics with statistical significance...")
    
    # Create summary dictionary
    summary = {
        'NYT': {
            'overall': {
                'mean': nyt_sentiments['compound_score'].mean(),
                'std': nyt_sentiments['compound_score'].std(),
                'median': nyt_sentiments['compound_score'].median(),
                'positive_ratio': (nyt_sentiments['sentiment'] == 'positive').mean(),
                'neutral_ratio': (nyt_sentiments['sentiment'] == 'neutral').mean(),
                'negative_ratio': (nyt_sentiments['sentiment'] == 'negative').mean(),
            }
        },
        'Peoples_Daily': {
            'overall': {
                'mean': pd_sentiments['compound_score'].mean(),
                'std': pd_sentiments['compound_score'].std(),
                'median': pd_sentiments['compound_score'].median(),
                'positive_ratio': (pd_sentiments['sentiment'] == 'positive').mean(),
                'neutral_ratio': (pd_sentiments['sentiment'] == 'neutral').mean(),
                'negative_ratio': (pd_sentiments['sentiment'] == 'negative').mean(),
            }
        }
    }
    
    # Add statistics for each context type
    context_types = ['medal', 'usa', 'china', 'usa_medal', 'china_medal', 'success', 'failure', 
                     'gold_medal', 'general_medal']
    
    for context in context_types:
        nyt_context = nyt_sentiments[nyt_sentiments['context_type'] == context]
        pd_context = pd_sentiments[pd_sentiments['context_type'] == context]
        
        if not nyt_context.empty:
            summary['NYT'][context] = {
                'mean': nyt_context['compound_score'].mean(),
                'std': nyt_context['compound_score'].std(),
                'median': nyt_context['compound_score'].median(),
                'positive_ratio': (nyt_context['sentiment'] == 'positive').mean(),
                'neutral_ratio': (nyt_context['sentiment'] == 'neutral').mean(),
                'negative_ratio': (nyt_context['sentiment'] == 'negative').mean(),
                'count': len(nyt_context)
            }
        
        if not pd_context.empty:
            summary['Peoples_Daily'][context] = {
                'mean': pd_context['compound_score'].mean(),
                'std': pd_context['compound_score'].std(),
                'median': pd_context['compound_score'].median(),
                'positive_ratio': (pd_context['sentiment'] == 'positive').mean(),
                'neutral_ratio': (pd_context['sentiment'] == 'neutral').mean(),
                'negative_ratio': (pd_context['sentiment'] == 'negative').mean(),
                'count': len(pd_context)
            }
    
    # Add Olympic vs non-Olympic years
    for source, df in [('NYT', nyt_sentiments), ('Peoples_Daily', pd_sentiments)]:
        olympic_years_df = df[df['is_olympic_year'] == True]
        non_olympic_years_df = df[df['is_olympic_year'] == False]
        
        summary[source]['olympic_years'] = {
            'mean': olympic_years_df['compound_score'].mean(),
            'std': olympic_years_df['compound_score'].std(),
            'positive_ratio': (olympic_years_df['sentiment'] == 'positive').mean(),
            'count': len(olympic_years_df)
        }
        
        summary[source]['non_olympic_years'] = {
            'mean': non_olympic_years_df['compound_score'].mean(),
            'std': non_olympic_years_df['compound_score'].std(),
            'positive_ratio': (non_olympic_years_df['sentiment'] == 'positive').mean(),
            'count': len(non_olympic_years_df)
        }
    
    # Create a picklable version of statistical test results
    # Convert unpicklable objects to simple values
    picklable_tests = {}
    if statistical_tests:
        for test_name, test_results in statistical_tests.items():
            picklable_tests[test_name] = {}
            for key, value in test_results.items():
                if isinstance(value, dict):
                    picklable_tests[test_name][key] = {}
                    for sub_key, sub_val in value.items():
                        if isinstance(sub_val, tuple) and len(sub_val) == 2:
                            # Convert tuple of stats to simple values
                            picklable_tests[test_name][key][sub_key] = {
                                'stat_value': float(sub_val[0]),
                                'p_value': float(sub_val[1])
                            }
                        else:
                            # Just store simple values
                            try:
                                # Try to make it a simple Python type
                                picklable_tests[test_name][key][sub_key] = float(sub_val) if isinstance(sub_val, (float, int)) else str(sub_val)
                            except:
                                # If conversion fails, just use a string representation
                                picklable_tests[test_name][key][sub_key] = f"Unpicklable: {type(sub_val)}"
                else:
                    # Handle non-dict values
                    try:
                        picklable_tests[test_name][key] = float(value) if isinstance(value, (float, int)) else str(value)
                    except:
                        picklable_tests[test_name][key] = f"Unpicklable: {type(value)}"
        
        # Add to summary
        summary['statistical_tests'] = picklable_tests
    
    # Save summary to file
    with open(os.path.join(OUTPUT_PATH, 'sentiment_summary_statistics.pkl'), 'wb') as f:
        pickle.dump(summary, f)
    
    # Create a more readable text version
    with open(os.path.join(OUTPUT_PATH, 'sentiment_summary_statistics.txt'), 'w') as f:
        f.write("Sentiment Analysis Summary Statistics\n")
        f.write("===================================\n\n")
        
        for source in ['NYT', 'Peoples_Daily']:
            f.write(f"{source} Statistics:\n")
            f.write("-----------------\n")
            
            for context, stats in summary[source].items():
                f.write(f"  {context.replace('_', ' ').title()}:\n")
                for stat_name, stat_value in stats.items():
                    if isinstance(stat_value, float):
                        f.write(f"    {stat_name}: {stat_value:.4f}\n")
                    else:
                        f.write(f"    {stat_name}: {stat_value}\n")
                f.write("\n")
            
            f.write("\n")
        
        # Add statistical test results
        if statistical_tests:
            f.write("Statistical Test Results:\n")
            f.write("------------------------\n")
            
            for test_name, test_results in picklable_tests.items():
                f.write(f"  {test_name}:\n")
                for key, value in test_results.items():
                    if isinstance(value, dict):
                        f.write(f"    {key}:\n")
                        for sub_key, sub_val in value.items():
                            if isinstance(sub_val, dict) and 'stat_value' in sub_val and 'p_value' in sub_val:
                                f.write(f"      {sub_key}: stat={sub_val['stat_value']:.4f}, p={sub_val['p_value']:.4f}\n")
                            else:
                                f.write(f"      {sub_key}: {sub_val}\n")
                    else:
                        f.write(f"    {key}: {value}\n")
                f.write("\n")
    
    print(f"Summary statistics saved to {OUTPUT_PATH}")
    return summary

# -----------------------------------------------------------------------
# 5. Main Functions
# -----------------------------------------------------------------------

def combine_extracted_contexts(contexts_dfs, medal_contexts_df):
    """Combine all extracted contexts for analysis."""
    combined_dfs = []
    
    # Add contexts from articles
    for key, df in contexts_dfs.items():
        if not df.empty:
            combined_dfs.append(df)
    
    # Add medal contexts from previous analysis
    if not medal_contexts_df.empty:
        combined_dfs.append(medal_contexts_df)
    
    # Combine all
    if combined_dfs:
        return pd.concat(combined_dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def main():
    """Main execution function."""
    start_time = time.time()
    
    # 1. Load data
    nyt_df, pd_df, nyt_medal_results, pd_medal_results, nyt_medal_contexts, pd_medal_contexts = load_datasets()
    
    # 2. Setup sentiment model
    # Check available GPU count
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    
    if gpu_count >= 2:
        # Set up models on two different GPUs
        device_0, tokenizer_0, model_0, sentiment_analyzer_0 = setup_sentiment_model(gpu_id=0)
        device_1, tokenizer_1, model_1, sentiment_analyzer_1 = setup_sentiment_model(gpu_id=1)
        use_parallel = True
        print("Using parallel processing with 2 GPUs")
    else:
        # Only one GPU available, set up just one model
        device_0, tokenizer_0, model_0, sentiment_analyzer_0 = setup_sentiment_model(gpu_id=0)
        tokenizer_1, model_1, device_1 = tokenizer_0, model_0, device_0
        sentiment_analyzer_1 = sentiment_analyzer_0
        use_parallel = False
        print("Only one GPU available, using sequential processing")
    
    # 3. Process the sentiment analysis in steps
    
    # Step 1: Extract contexts from articles
    nyt_contexts_path = os.path.join(OUTPUT_PATH, "nyt_extracted_contexts.pkl")
    pd_contexts_path = os.path.join(OUTPUT_PATH, "pd_extracted_contexts.pkl")
    
    if os.path.exists(nyt_contexts_path) and os.path.exists(pd_contexts_path):
        print("Loading previously extracted contexts...")
        with open(nyt_contexts_path, 'rb') as f:
            nyt_contexts_dfs = pickle.load(f)
        with open(pd_contexts_path, 'rb') as f:
            pd_contexts_dfs = pickle.load(f)
    else:
        # Extract contexts from articles
        # For initial development/testing, consider using a smaller sample
        # nyt_sample = nyt_df.sample(min(5000, len(nyt_df)), random_state=42)
        # pd_sample = pd_df.sample(min(5000, len(pd_df)), random_state=42)
        
        nyt_contexts_dfs = extract_contexts_from_articles(nyt_df, is_english=True)
        pd_contexts_dfs = extract_contexts_from_articles(pd_df, is_english=False)
        
        # Save extracted contexts
        with open(nyt_contexts_path, 'wb') as f:
            pickle.dump(nyt_contexts_dfs, f)
        with open(pd_contexts_path, 'wb') as f:
            pickle.dump(pd_contexts_dfs, f)
    
    # Step 2: Process medal contexts from previous analysis
    nyt_medal_contexts_df = extract_medal_contexts(nyt_medal_contexts, is_english=True)
    pd_medal_contexts_df = extract_medal_contexts(pd_medal_contexts, is_english=False)
    
    # Step 3: Combine contexts for analysis
    nyt_all_contexts = combine_extracted_contexts(nyt_contexts_dfs, nyt_medal_contexts_df)
    pd_all_contexts = combine_extracted_contexts(pd_contexts_dfs, pd_medal_contexts_df)
    
    print(f"Total NYT contexts for analysis: {len(nyt_all_contexts)}")
    print(f"Total People's Daily contexts for analysis: {len(pd_all_contexts)}")
    
    # Step 4: Analyze sentiment
    nyt_sentiments_path = os.path.join(OUTPUT_PATH, "nyt_sentiments.pkl")
    pd_sentiments_path = os.path.join(OUTPUT_PATH, "pd_sentiments.pkl")
    
    if os.path.exists(nyt_sentiments_path) and os.path.exists(pd_sentiments_path):
        print("Loading previously analyzed sentiments...")
        nyt_sentiments = pd.read_pickle(nyt_sentiments_path)
        pd_sentiments = pd.read_pickle(pd_sentiments_path)
    else:
        # Use multi-GPU parallel processing
        if use_parallel:
            # Create and start threads
            nyt_thread = threading.Thread(
                target=process_nyt_sentiments, 
                args=(nyt_all_contexts, tokenizer_0, model_0, device_0, OUTPUT_PATH)
            )
            
            pd_thread = threading.Thread(
                target=process_pd_sentiments,
                args=(pd_all_contexts, tokenizer_1, model_1, device_1, OUTPUT_PATH)
            )
            
            nyt_thread.start()
            pd_thread.start()
            
            # Wait for both threads to complete
            nyt_thread.join()
            pd_thread.join()
            
            # Load saved results
            nyt_sentiments = pd.read_pickle(nyt_sentiments_path)
            pd_sentiments = pd.read_pickle(pd_sentiments_path)
        else:
            # Sequential processing (if only one GPU)
            print("Processing sentiments sequentially...")
            nyt_sentiments = process_context_sentiments(
                nyt_all_contexts, 
                tokenizer_0, 
                model_0, 
                device_0,
                batch_size=512,
                num_workers=MAX_CPUS  # Use all CPU cores for sequential processing
            )
            nyt_sentiments.to_pickle(nyt_sentiments_path)
            
            pd_sentiments = process_context_sentiments(
                pd_all_contexts, 
                tokenizer_1, 
                model_1, 
                device_1,
                batch_size=512,
                num_workers=MAX_CPUS
            )
            pd_sentiments.to_pickle(pd_sentiments_path)
    
    # Step 5: Generate visualizations and analysis
    
    # Statistical test results container
    statistical_tests = {}
    
    # Basic sentiment distributions (now separated for each publication)
    plot_sentiment_distribution_separate(nyt_sentiments, pd_sentiments, 'all')
    plot_sentiment_distribution_separate(
        nyt_sentiments[nyt_sentiments['context_type'] == 'medal'], 
        pd_sentiments[pd_sentiments['context_type'] == 'medal'], 
        'medal'
    )
    
    # Time series analysis (keeping original combined format for trend comparison)
    plot_sentiment_time_series(nyt_sentiments, pd_sentiments, 'all')
    plot_sentiment_time_series(
        nyt_sentiments[nyt_sentiments['context_type'] == 'medal'], 
        pd_sentiments[pd_sentiments['context_type'] == 'medal'], 
        'medal'
    )
    
    # Comparative analyses with separated charts and statistical tests
    home_rival_results = plot_home_vs_rival_sentiment_separate(nyt_sentiments, pd_sentiments)
    statistical_tests['home_vs_rival'] = home_rival_results
    
    success_failure_results = plot_success_vs_failure_sentiment_separate(nyt_sentiments, pd_sentiments)
    statistical_tests['success_vs_failure'] = success_failure_results
    
    beijing_results = plot_beijing_olympics_impact_separate(nyt_sentiments, pd_sentiments)
    statistical_tests['beijing_olympics'] = beijing_results
    
    gold_vs_general_results = analyze_gold_vs_general_sentiment_separate(nyt_sentiments, pd_sentiments)
    statistical_tests['gold_vs_general'] = gold_vs_general_results
    
    # Generate summary statistics with statistical test results
    summary = generate_sentiment_summary_statistics(nyt_sentiments, pd_sentiments, statistical_tests)
    
    # Print completion information
    total_time = time.time() - start_time
    print(f"\nSentiment analysis completed in {total_time:.2f} seconds")
    print(f"Results saved to {OUTPUT_PATH}")
    
    # Print key findings
    print("\nKey Findings:")
    print("-" * 40)
    
    # Overall sentiment
    nyt_overall = summary['NYT']['overall']['mean']
    pd_overall = summary['Peoples_Daily']['overall']['mean']
    print(f"Overall sentiment (NYT): {nyt_overall:.4f}")
    print(f"Overall sentiment (People's Daily): {pd_overall:.4f}")
    
    # Medal sentiment
    if 'medal' in summary['NYT'] and 'medal' in summary['Peoples_Daily']:
        nyt_medal = summary['NYT']['medal']['mean']
        pd_medal = summary['Peoples_Daily']['medal']['mean']
        print(f"Medal contexts sentiment (NYT): {nyt_medal:.4f}")
        print(f"Medal contexts sentiment (People's Daily): {pd_medal:.4f}")
    
    # Success vs. Failure
    if 'success' in summary['NYT'] and 'failure' in summary['NYT']:
        nyt_gap = summary['NYT']['success']['mean'] - summary['NYT']['failure']['mean']
        print(f"Success-Failure sentiment gap (NYT): {nyt_gap:.4f}")
        print(f"p-value: {statistical_tests['success_vs_failure']['nyt']['p_value']:.4f}")
    
    if 'success' in summary['Peoples_Daily'] and 'failure' in summary['Peoples_Daily']:
        pd_gap = summary['Peoples_Daily']['success']['mean'] - summary['Peoples_Daily']['failure']['mean']
        print(f"Success-Failure sentiment gap (People's Daily): {pd_gap:.4f}")
        print(f"p-value: {statistical_tests['success_vs_failure']['pd']['p_value']:.4f}")
    
    # Beijing Olympics impact
    print("\nBeijing Olympics Impact (ANOVA p-values):")
    print(f"NYT: {statistical_tests['beijing_olympics']['nyt']['anova'][1]:.4f}")
    print(f"People's Daily: {statistical_tests['beijing_olympics']['pd']['anova'][1]:.4f}")
    
    print("-" * 40)

if __name__ == "__main__":
    main()