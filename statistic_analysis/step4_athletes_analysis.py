# Project 4: Athlete Mention Analysis - Olympic Individuals and National Attribution
# =================================================================
# This script analyzes how NYT and People's Daily frame Olympic athletes
# in relation to their nations, examining the individual vs. collective attribution
# patterns in Olympic coverage across media cultures.

import os
import pandas as pd
import numpy as np
import re
import ast
import jieba
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import time
import multiprocessing as mp
from functools import partial
import math
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings("ignore")

# Set paths for Linux environment
DATA_PATH = "/U_PZL2021KF0012/hx/EPF/History_and_digital/project_data/processed_data"
ATHLETE_DATA_PATH = "/U_PZL2021KF0012/hx/EPF/History_and_digital/project_data/athlete_data"
OUTPUT_PATH = "/U_PZL2021KF0012/hx/EPF/History_and_digital/project_data/athlete_analysis"

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

# Initialize Jieba for Chinese text segmentation
jieba.initialize()

# Maximum CPU cores for parallel processing
MAX_CPUS = min(mp.cpu_count(), 128)
print(f"Setting up parallel processing with {MAX_CPUS} CPU cores")

# Define Olympic years for reference
summer_olympic_years = [1980, 1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012]
winter_olympic_years = [1980, 1984, 1988, 1992, 1994, 1998, 2002, 2006, 2010, 2014]
olympic_years = sorted(list(set(summer_olympic_years + winter_olympic_years)))
beijing_olympics_year = 2008

# -----------------------------------------------------------------------
# 1. Data Loading Functions
# -----------------------------------------------------------------------

def load_datasets():
    """Load previously processed article datasets."""
    print("Loading article datasets...")
    
    # Load standardized article datasets
    nyt_path = os.path.join(DATA_PATH, "nyt_standardized.pkl")
    pd_path = os.path.join(DATA_PATH, "people_daily_standardized.pkl")
    
    nyt_df = pd.read_pickle(nyt_path)
    pd_df = pd.read_pickle(pd_path)
    
    # Mark Olympic years
    nyt_df['is_olympic_year'] = nyt_df['year'].isin(olympic_years)
    pd_df['is_olympic_year'] = pd_df['year'].isin(olympic_years)
    
    print(f"NYT dataset: {len(nyt_df)} articles")
    print(f"People's Daily dataset: {len(pd_df)} articles")
    
    return nyt_df, pd_df

def load_athlete_data():
    """Load and preprocess athlete data with name variations."""
    print("Loading athlete data...")
    
    # Load athlete data
    usa_athletes_path = os.path.join(ATHLETE_DATA_PATH, "usa_athletes_processed.csv")
    chinese_athletes_path = os.path.join(ATHLETE_DATA_PATH, "chinese_athletes_processed.csv")
    
    usa_athletes_df = pd.read_csv(usa_athletes_path, encoding='utf-8-sig')
    chinese_athletes_df = pd.read_csv(chinese_athletes_path, encoding='utf-8-sig')
    
    # Process USA athletes (with Chinese name variations)
    usa_athletes = {}
    for _, row in usa_athletes_df.iterrows():
        # Convert string representation of list to actual list
        chinese_variants = ast.literal_eval(row['Possible_Chinese_Names']) if isinstance(row['Possible_Chinese_Names'], str) else []
        
        # Add original English name to the variations
        english_name = row['Name']
        athlete_year = row['Year']
        english_variants = [english_name]  # Start with the original name
        english_dict_name = str(athlete_year) + '_' + english_name 
        
        # Add name variations if available
        if 'Possible_variations' in row and pd.notna(row['Possible_variations']):
            english_variants.extend(ast.literal_eval(row['Possible_variations']))
        
        # Create entry in dictionary
        usa_athletes[english_dict_name] = {
            'year': int(row['Year']),
            'season': row['Season'],
            'medal_count': int(row['Medal_Count']),
            'nationality': 'USA',
            'english_variants': english_variants,
            'chinese_variants': chinese_variants
        }
    
    # Process Chinese athletes (with English name variations)
    chinese_athletes = {}
    for _, row in chinese_athletes_df.iterrows():
        # Convert string representation of list to actual list
        english_variants = ast.literal_eval(row['Possible_English_Names']) if isinstance(row['Possible_English_Names'], str) else []
        
        # Add original Chinese name to the variations
        chinese_name = row['Name']
        athlete_year = row['Year']
        chinese_variants = [chinese_name]  # Start with the original name
        chinese_dict_name = str(athlete_year) + '_' + chinese_name 
        
        # Add name variations if available
        if 'Possible_variations' in row and pd.notna(row['Possible_variations']):
            chinese_variants.extend(ast.literal_eval(row['Possible_variations']))
        
        # Create entry in dictionary
        chinese_athletes[chinese_dict_name] = {
            'year': int(row['Year']),
            'season': row['Season'],
            'medal_count': int(row['Medal_Count']),
            'nationality': 'China',
            'english_variants': english_variants,
            'chinese_variants': chinese_variants
        }
    
    print(f"Loaded {len(usa_athletes)} USA athletes and {len(chinese_athletes)} Chinese athletes")
    return usa_athletes, chinese_athletes

# -----------------------------------------------------------------------
# 2. Context Extraction Functions
# -----------------------------------------------------------------------

def get_eligible_athletes(athletes_dict, article_year, time_window=4):
    """
    Get athletes that are eligible for mention in an article based on its year.
    Only includes athletes from up to 'time_window' years before the article.
    
    Args:
        athletes_dict: Dictionary of athletes
        article_year: Year of the article
        time_window: How many years back to consider (default: 4 years)
    
    Returns:
        Dictionary of eligible athletes
    """
    eligible_athletes = {}
    
    for name, info in athletes_dict.items():
        # Check if athlete's Olympic year is within the time window
        if info['year'] <= article_year and info['year'] >= article_year - time_window:
            eligible_athletes[name] = info
    
    return eligible_athletes

def extract_context(text, match_start, match_end, context_window_chars=200):
    """Extract a context window around a matched name in text."""
    # Find sentence boundaries if possible
    # First get a wider window to ensure we have full sentences
    search_start = max(0, match_start - context_window_chars*2)
    search_end = min(len(text), match_end + context_window_chars*2)
    search_text = text[search_start:search_end]
    
    # Find the sentence containing the match
    match_in_search = match_start - search_start
    
    # Try to find sentence boundaries
    sentence_start = search_text.rfind('. ', 0, match_in_search)
    if sentence_start == -1:
        sentence_start = search_text.rfind('! ', 0, match_in_search)
    if sentence_start == -1:
        sentence_start = search_text.rfind('? ', 0, match_in_search)
    if sentence_start == -1:
        sentence_start = max(0, match_in_search - context_window_chars)
    else:
        # Move past the period and space
        sentence_start += 2
    
    sentence_end = search_text.find('. ', match_in_search)
    if sentence_end == -1:
        sentence_end = search_text.find('! ', match_in_search)
    if sentence_end == -1:
        sentence_end = search_text.find('? ', match_in_search)
    if sentence_end == -1:
        sentence_end = min(len(search_text), match_in_search + context_window_chars)
    else:
        # Include the period
        sentence_end += 1
    
    # Extract the sentence
    context = search_text[sentence_start:sentence_end].strip()
    
    # If context is too short, take a wider window
    if len(context) < 40:  # Minimum context length
        context = text[max(0, match_start - context_window_chars):min(len(text), match_end + context_window_chars)]
    
    return context

def extract_athlete_mentions_batch(articles_batch, athletes_dict, is_english=True, time_window=4):
    """
    Extract mentions of athletes from a batch of articles.
    
    Args:
        articles_batch: Batch of articles to process
        athletes_dict: Dictionary of athletes with their name variations
        is_english: Whether the articles are in English or Chinese
        time_window: How many years back to consider for athletes
    
    Returns:
        List of dicts with article and athlete mention information
    """
    mentions = []
    
    for _, article in articles_batch.iterrows():
        content = article['content']
        article_year = article['year']
        
        if not isinstance(content, str) or not content.strip():
            continue
        
        # Get athletes eligible for mention in this article
        eligible_athletes = get_eligible_athletes(athletes_dict, article_year, time_window)
        
        for original_name, athlete_info in eligible_athletes.items():
            # Use appropriate language variants
            name_variants = athlete_info['english_variants'] if is_english else athlete_info['chinese_variants']
            
            # Check for each name variant
            for name in name_variants:
                if not name or not isinstance(name, str):
                    continue
                
                if is_english:
                    # For English, ensure we're matching full words
                    pattern = r'\b' + re.escape(name) + r'\b'
                    flags = re.IGNORECASE
                else:
                    # For Chinese, direct matching is fine
                    pattern = re.escape(name)
                    flags = 0
                
                for match in re.finditer(pattern, content, flags):
                    # Extract context around the mention
                    context = extract_context(content, match.start(), match.end())
                    
                    # Record the mention
                    mentions.append({
                        'article_id': article['original_id'],
                        'article_year': article_year,
                        'article_month': article['month'],
                        'is_olympic_year': article['is_olympic_year'],
                        'original_athlete_name': original_name,
                        'matched_name': name,
                        'nationality': athlete_info['nationality'],
                        'medal_count': athlete_info['medal_count'],
                        'athlete_olympic_year': athlete_info['year'],
                        'athlete_season': athlete_info['season'],
                        'context': context,
                        # Track the time difference
                        'years_since_olympics': article_year - athlete_info['year']
                    })
    
    return mentions

def process_articles_parallel(df, athletes_dict, is_english=True, time_window=4):
    """Process articles in parallel to extract athlete mentions."""
    print(f"Processing {'English' if is_english else 'Chinese'} articles for athlete mentions...")
    
    # Split the dataframe into batches for parallel processing
    num_articles = len(df)
    batch_size = math.ceil(num_articles / MAX_CPUS)
    batches = [df.iloc[i:i+batch_size] for i in range(0, num_articles, batch_size)]
    
    print(f"Processing {num_articles} articles in {len(batches)} batches")
    
    # Set up the partial function for parallel processing
    process_batch = partial(
        extract_athlete_mentions_batch,
        athletes_dict=athletes_dict,
        is_english=is_english,
        time_window=time_window
    )
    
    # Process in parallel
    with mp.Pool(processes=MAX_CPUS) as pool:
        results = list(tqdm(
            pool.imap(process_batch, batches),
            total=len(batches),
            desc=f"Extracting {'English' if is_english else 'Chinese'} athlete mentions"
        ))
    
    # Flatten results
    all_mentions = [mention for batch_result in results for mention in batch_result]
    print(f"Found {len(all_mentions)} athlete mentions")
    
    return all_mentions

# -----------------------------------------------------------------------
# 3. Analysis Functions
# -----------------------------------------------------------------------


INDIVIDUAL_COLLECTIVE_MARKERS = {
    'english': {
        'individual_pronouns': ['he', 'she', 'his', 'her', 'him', 'himself', 'herself', 'i', 'my', 'me', 'mine'],
        
        'collective_pronouns': [
            # 美国相关
            'we', 'our', 'us', 'ourselves', 'team', 'american', 'americans', 'united states', 
            'team usa', 'u.s.', 'u.s.a.', 'america',
            # 中国相关 (新增)
            'chinese', 'china', 'team china', 'prc', 'people\'s republic', 'chinese team'
        ],
        
        'individual_verbs': ['won', 'achieved', 'earned', 'claimed', 'took', 'grabbed', 'secured', 'captured', 
                            'dominated', 'triumphed', 'succeeded', 'performed', 'competed', 'finished'],
        
        'collective_phrases': [
            # 美国相关
            'represent', 'representing', 'for the united states', 'for america', 'for the u.s.', 
            'for his country', 'for her country', 'for the team', 'for the nation', 
            'bringing honor to', 'medal for the u.s.', 'american pride', 'national pride',
            'american hero', 'team victory', 'american victory',
            # 中国相关 (新增)
            'for china', 'for the people\'s republic', 'medal for china', 'chinese pride', 
            'representing china', 'china\'s honor', 'honor for china', 'chinese team victory',
            'china\'s victory', 'glory for china', 'china\'s glory', 'chinese national team'
        ]
    },
    
    'chinese': {
        'individual_pronouns': ['他', '她', '他的', '她的', '其', '自己', '本人', '个人', '该选手', '这位选手',
                               '这名选手', '这位运动员', '这名运动员'],
        
        'collective_pronouns': [
            # 中国相关
            '我们', '我国', '中国', '国家', '祖国', '队', '中国队', '国家队', '中华', '我们的', 
            '我国的', '中国的', '祖国的', '队的',
            # 美国相关 (新增)
            '美国', '美国队', '美国人', '美利坚', '美', '美方', '美方队伍', '美国国家队'
        ],
        
        'individual_verbs': ['获得', '夺得', '赢得', '拿到', '摘得', '取得', '斩获', '创造', '打破', '刷新', 
                            '完成', '击败', '超越', '战胜'],
        
        'collective_phrases': [
            # 中国相关
            '为国争光', '为国家', '为祖国', '为中国', '中国骄傲', '国家荣誉', '中国自豪', 
            '国人自豪', '全国欢庆', '举国欢腾', '代表中国', '国家队的骄傲', '为中华民族', 
            '为人民', '体现国家实力', '中国力量', '中国精神', '为团队', '集体荣誉', 
            '为中国队', '为国家队',
            # 美国相关 (新增)
            '为美国争光', '为美国', '美国骄傲', '美国荣誉', '美国自豪', '代表美国', 
            '美国队的骄傲', '为美国人民', '体现美国实力', '美国力量', '美国精神', 
            '为美国队', '美国集体荣誉', '美国团队', '美国的胜利', '美国的荣耀'
        ]
    }
}

def analyze_individual_vs_national_framing(mentions_df, is_english=True):
    """
    Analyze individual vs. national framing in athlete mentions.
    
    This function analyzes each athlete mention context to determine whether
    the athlete is framed more as an individual or as a representative of their nation.
    """
    print(f"Analyzing {'English' if is_english else 'Chinese'} mentions for individual vs. national framing...")
    
    # Get the appropriate markers based on language
    lang = 'english' if is_english else 'chinese'
    markers = INDIVIDUAL_COLLECTIVE_MARKERS[lang]
    
    # Initialize results list
    results = []
    
    # Process each mention
    for _, mention in tqdm(mentions_df.iterrows(), total=len(mentions_df), 
                          desc="Analyzing individual vs. national framing"):
        context = mention['context'].lower() if is_english else mention['context']
        
        # Count individual and collective pronouns
        individual_pronoun_count = sum(context.count(pronoun) for pronoun in markers['individual_pronouns'])
        collective_pronoun_count = sum(context.count(pronoun) for pronoun in markers['collective_pronouns'])
        
        # Count individual verbs
        individual_verb_count = sum(context.count(verb) for verb in markers['individual_verbs'])
        
        # Count collective phrases
        collective_phrase_count = sum(context.count(phrase) for phrase in markers['collective_phrases'])
        
        # Calculate individual vs. national attribution scores
        total_individual_markers = individual_pronoun_count + individual_verb_count
        total_collective_markers = collective_pronoun_count + collective_phrase_count
        
        # Avoid division by zero
        total_markers = total_individual_markers + total_collective_markers
        individual_ratio = total_individual_markers / total_markers if total_markers > 0 else 0.5
        
        # For English, check for specific sentence patterns using regex
        # These are more complex constructions that indicate attribution style
        individual_patterns = 0
        collective_patterns = 0
        
        if is_english:
            # Patterns indicating individual achievement
            individual_pattern_regexes = [
                r'\b(he|she)\s+\w+ed\b',  # "he won", "she earned"
                r'\bhis|her\b\s+\w+',  # "his victory", "her medal"
                r'\b(the athlete|the swimmer|the gymnast)\b',  # Focus on athlete's role
                r'\bindividual\s+\w+\b'  # "individual achievement"
            ]
            
            # Patterns indicating national/collective achievement
            collective_pattern_regexes = [
                r'\bfor\s+(the\s+)?(united states|america|u\.s\.|team)\b',  # "for the United States"
                r'\bteam\s+\w+\b',  # "team effort", "team victory"
                r'\bnational\s+\w+\b',  # "national pride"
                r'\b(united states|american|u\.s\.)\s+\w+\b'  # "American victory"
            ]
            
            for pattern in individual_pattern_regexes:
                individual_patterns += len(re.findall(pattern, context))
            
            for pattern in collective_pattern_regexes:
                collective_patterns += len(re.findall(pattern, context))
        else:
            # Chinese patterns indicating individual achievement
            individual_pattern_regexes = [
                r'他|她\s*\w+了',  # "他获得了", "她赢得了"
                r'这名选手|这位选手|这名运动员|这位运动员',  # Focus on athlete
                r'个人\s*\w+',  # "个人成就"
                r'自己的\s*\w+'  # "自己的努力"
            ]
            
            # Chinese patterns indicating national/collective achievement
            collective_pattern_regexes = [
                r'为\s*(中国|国家|祖国|人民)\s*\w+',  # "为中国争光"
                r'(中国|国家|祖国)\s*的\s*\w+',  # "中国的骄傲"
                r'(全国|举国|国人)\s*\w+',  # "全国欢庆"
                r'代表\s*(中国|国家|祖国)'  # "代表中国"
            ]
            
            for pattern in individual_pattern_regexes:
                individual_patterns += len(re.findall(pattern, context))
            
            for pattern in collective_pattern_regexes:
                collective_patterns += len(re.findall(pattern, context))
        
        # Adjust the ratio based on pattern matches
        pattern_total = individual_patterns + collective_patterns
        if pattern_total > 0:
            pattern_ratio = individual_patterns / pattern_total
            # Weight the final ratio (75% markers, 25% patterns)
            individual_ratio = 0.75 * individual_ratio + 0.25 * pattern_ratio
        
        # Create result entry
        result = {
            'article_id': mention['article_id'],
            'article_year': mention['article_year'],
            'athlete_name': mention['original_athlete_name'],
            'nationality': mention['nationality'],
            'medal_count': mention['medal_count'],
            'athlete_olympic_year': mention['athlete_olympic_year'],
            'years_since_olympics': mention['years_since_olympics'],
            'individual_pronoun_count': individual_pronoun_count,
            'collective_pronoun_count': collective_pronoun_count,
            'individual_verb_count': individual_verb_count,
            'collective_phrase_count': collective_phrase_count,
            'individual_patterns': individual_patterns,
            'collective_patterns': collective_patterns,
            'individual_attribution_score': total_individual_markers,
            'collective_attribution_score': total_collective_markers,
            'individual_ratio': individual_ratio,  # Higher means more individual-focused
            'context': mention['context']
        }
        
        results.append(result)
    
    return pd.DataFrame(results)

def calculate_aggregate_statistics(framing_df, groupby_col='article_year'):
    """
    Calculate aggregate statistics for individual vs. national framing.
    
    Args:
        framing_df: DataFrame with framing analysis results
        groupby_col: Column to group by for aggregation (e.g., 'article_year', 'nationality')
    
    Returns:
        DataFrame with aggregated statistics
    """
    # Group by the specified column and calculate mean ratios
    agg_stats = framing_df.groupby(groupby_col).agg({
        'individual_ratio': ['mean', 'std', 'count'],
        'individual_attribution_score': 'sum',
        'collective_attribution_score': 'sum'
    }).reset_index()
    
    # Flatten the multi-level columns
    agg_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in agg_stats.columns]
    
    # Calculate overall attribution ratio from summed scores
    agg_stats['overall_individual_ratio'] = (
        agg_stats['individual_attribution_score_sum'] / 
        (agg_stats['individual_attribution_score_sum'] + agg_stats['collective_attribution_score_sum'])
    )
    
    return agg_stats

def analyze_nationality_bias(framing_df, is_english=True):
    """
    Analyze how nationality affects individual vs. national framing.
    
    Args:
        framing_df: DataFrame with framing analysis results
        is_english: Whether the data is from English or Chinese sources
    
    Returns:
        DataFrame with nationality bias statistics
    """
    # Define own country and rival country based on language
    own_country = 'USA' if is_english else 'China'
    rival_country = 'China' if is_english else 'USA'
    
    # Filter for own country and rival country
    own_country_data = framing_df[framing_df['nationality'] == own_country]
    rival_country_data = framing_df[framing_df['nationality'] == rival_country]
    
    # Calculate mean individual ratios
    own_country_ratio = own_country_data['individual_ratio'].mean()
    rival_country_ratio = rival_country_data['individual_ratio'].mean()
    
    # Calculate standard deviations
    own_country_std = own_country_data['individual_ratio'].std()
    rival_country_std = rival_country_data['individual_ratio'].std()
    
    # Calculate counts
    own_country_count = len(own_country_data)
    rival_country_count = len(rival_country_data)
    
    # Create result DataFrame
    result = pd.DataFrame({
        'country': [own_country, rival_country],
        'mean_individual_ratio': [own_country_ratio, rival_country_ratio],
        'std_individual_ratio': [own_country_std, rival_country_std],
        'count': [own_country_count, rival_country_count],
        'is_own_country': [True, False]
    })
    
    return result

def analyze_athlete_prominence(framing_df):
    """
    Analyze how athlete prominence (medal count) affects framing.
    
    Args:
        framing_df: DataFrame with framing analysis results
    
    Returns:
        DataFrame with statistics grouped by medal count
    """
    # Create medal count categories
    framing_df['medal_category'] = pd.cut(
        framing_df['medal_count'],
        bins=[0, 1, 3, float('inf')],
        labels=['1 medal', '2-3 medals', '4+ medals']
    )
    
    # Group by medal category and calculate statistics
    medal_stats = framing_df.groupby(['nationality', 'medal_category']).agg({
        'individual_ratio': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten the multi-level columns
    medal_stats.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in medal_stats.columns]
    
    return medal_stats

def analyze_beijing_olympics_impact(nyt_framing, pd_framing):
    """
    Analyze the impact of the 2008 Beijing Olympics on framing.
    
    Args:
        nyt_framing: DataFrame with NYT framing analysis results
        pd_framing: DataFrame with People's Daily framing analysis results
    
    Returns:
        Dictionary with pre, during, and post Beijing Olympics statistics
    """
    # Define time periods
    pre_beijing = [2004, 2005, 2006, 2007]
    during_beijing = [2008]
    post_beijing = [2009, 2010, 2011, 2012]
    
    # Filter NYT data for each period
    nyt_pre = nyt_framing[nyt_framing['article_year'].isin(pre_beijing)]
    nyt_during = nyt_framing[nyt_framing['article_year'].isin(during_beijing)]
    nyt_post = nyt_framing[nyt_framing['article_year'].isin(post_beijing)]
    
    # Filter People's Daily data for each period
    pd_pre = pd_framing[pd_framing['article_year'].isin(pre_beijing)]
    pd_during = pd_framing[pd_framing['article_year'].isin(during_beijing)]
    pd_post = pd_framing[pd_framing['article_year'].isin(post_beijing)]
    
    # Calculate mean individual ratios for each period
    result = {
        'NYT': {
            'pre_beijing': {
                'mean': nyt_pre['individual_ratio'].mean(),
                'std': nyt_pre['individual_ratio'].std(),
                'count': len(nyt_pre)
            },
            'during_beijing': {
                'mean': nyt_during['individual_ratio'].mean(),
                'std': nyt_during['individual_ratio'].std(),
                'count': len(nyt_during)
            },
            'post_beijing': {
                'mean': nyt_post['individual_ratio'].mean(),
                'std': nyt_post['individual_ratio'].std(),
                'count': len(nyt_post)
            }
        },
        'PeoplesDaily': {
            'pre_beijing': {
                'mean': pd_pre['individual_ratio'].mean(),
                'std': pd_pre['individual_ratio'].std(),
                'count': len(pd_pre)
            },
            'during_beijing': {
                'mean': pd_during['individual_ratio'].mean(),
                'std': pd_during['individual_ratio'].std(),
                'count': len(pd_during)
            },
            'post_beijing': {
                'mean': pd_post['individual_ratio'].mean(),
                'std': pd_post['individual_ratio'].std(),
                'count': len(pd_post)
            }
        }
    }
    
    return result

# -----------------------------------------------------------------------
# 4. Visualization Functions
# -----------------------------------------------------------------------

def plot_individual_national_trends(nyt_yearly, pd_yearly):
    """
    Plot the trend of individual vs. national attribution over time.
    
    Args:
        nyt_yearly: Yearly statistics for NYT
        pd_yearly: Yearly statistics for People's Daily
    """
    plt.figure(figsize=(14, 7))
    
    # Plot NYT trend
    plt.plot(nyt_yearly['article_year'], nyt_yearly['individual_ratio_mean'], 
             marker='o', linewidth=2, label='NYT', color='#1f77b4')
    
    # Add shaded area for standard deviation
    plt.fill_between(
        nyt_yearly['article_year'],
        nyt_yearly['individual_ratio_mean'] - nyt_yearly['individual_ratio_std'],
        nyt_yearly['individual_ratio_mean'] + nyt_yearly['individual_ratio_std'],
        alpha=0.2, color='#1f77b4'
    )
    
    # Plot People's Daily trend
    plt.plot(pd_yearly['article_year'], pd_yearly['individual_ratio_mean'], 
             marker='s', linewidth=2, label='People\'s Daily', color='#ff7f0e')
    
    # Add shaded area for standard deviation
    plt.fill_between(
        pd_yearly['article_year'],
        pd_yearly['individual_ratio_mean'] - pd_yearly['individual_ratio_std'],
        pd_yearly['individual_ratio_mean'] + pd_yearly['individual_ratio_std'],
        alpha=0.2, color='#ff7f0e'
    )
    
    # Add vertical lines for Olympic years
    for year in olympic_years:
        if year >= min(nyt_yearly['article_year'].min(), pd_yearly['article_year'].min()) and \
           year <= max(nyt_yearly['article_year'].max(), pd_yearly['article_year'].max()):
            plt.axvline(x=year, color='gray', linestyle='--', alpha=0.5)
    
    # Add vertical line for Beijing Olympics
    plt.axvline(x=2008, color='red', linestyle='-', alpha=0.7)
    plt.text(2008, plt.ylim()[1]*0.95, 'Beijing\nOlympics', 
             ha='center', va='top', color='red', fontweight='bold')
    
    # Add labels and formatting
    plt.title('Individual vs. National Attribution in Olympic Athlete Coverage (1980-2015)', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Individual Attribution Ratio\n(Higher = More Individual Focus)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set y-axis to range from 0 to 1
    plt.ylim(0, 1)
    
    # Add a horizontal line at 0.5 to indicate equal attribution
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    plt.text(plt.xlim()[0], 0.51, 'Equal Attribution', fontsize=10, ha='left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'individual_national_attribution_trends.png'), dpi=300)
    plt.close()

def plot_nationality_comparison(nyt_nationality_bias, pd_nationality_bias):
    """
    Plot how athletes of different nationalities are framed.
    
    Args:
        nyt_nationality_bias: Nationality bias statistics for NYT
        pd_nationality_bias: Nationality bias statistics for People's Daily
    """
    plt.figure(figsize=(10, 6))
    
    # Set up positions for bars
    x = np.arange(2)  # Own Country, Rival Country
    width = 0.35
    
    # Extract data
    nyt_own = nyt_nationality_bias[nyt_nationality_bias['is_own_country']]['mean_individual_ratio'].values[0]
    nyt_rival = nyt_nationality_bias[~nyt_nationality_bias['is_own_country']]['mean_individual_ratio'].values[0]
    
    pd_own = pd_nationality_bias[pd_nationality_bias['is_own_country']]['mean_individual_ratio'].values[0]
    pd_rival = pd_nationality_bias[~pd_nationality_bias['is_own_country']]['mean_individual_ratio'].values[0]
    
    # Plot bars
    plt.bar(x - width/2, [nyt_own, nyt_rival], width, label='NYT', color='#1f77b4')
    plt.bar(x + width/2, [pd_own, pd_rival], width, label='People\'s Daily', color='#ff7f0e')
    
    # Add labels and formatting
    plt.title('Individual Attribution by Athlete Nationality', fontsize=16)
    plt.ylabel('Individual Attribution Ratio\n(Higher = More Individual Focus)', fontsize=12)
    plt.xticks(x, ['Own Country Athletes', 'Rival Country Athletes'])
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Set y-axis to range from 0 to 1
    plt.ylim(0, 1)
    
    # Add a horizontal line at 0.5 to indicate equal attribution
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for i, v in enumerate([nyt_own, nyt_rival]):
        plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
    
    for i, v in enumerate([pd_own, pd_rival]):
        plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'nationality_bias_comparison.png'), dpi=300)
    plt.close()

def plot_medal_count_comparison(nyt_medal_stats, pd_medal_stats):
    """
    Plot how medal count affects individual vs. national framing.
    
    Args:
        nyt_medal_stats: Medal count statistics for NYT
        pd_medal_stats: Medal count statistics for People's Daily
    """
    plt.figure(figsize=(12, 7))
    
    # Extract USA and China data
    nyt_usa = nyt_medal_stats[nyt_medal_stats['nationality'] == 'USA']
    nyt_china = nyt_medal_stats[nyt_medal_stats['nationality'] == 'China']
    
    pd_usa = pd_medal_stats[pd_medal_stats['nationality'] == 'USA']
    pd_china = pd_medal_stats[pd_medal_stats['nationality'] == 'China']
    
    # Set up positions for grouped bars
    medal_categories = ['1 medal', '2-3 medals', '4+ medals']
    x = np.arange(len(medal_categories))
    width = 0.2
    
    # Ensure all categories exist in each dataset
    for df in [nyt_usa, nyt_china, pd_usa, pd_china]:
        for category in medal_categories:
            if category not in df['medal_category'].values:
                # Add a row with NaN values
                new_row = pd.Series({
                    'nationality': df['nationality'].iloc[0] if len(df) > 0 else None,
                    'medal_category': category,
                    'individual_ratio_mean': np.nan,
                    'individual_ratio_std': np.nan,
                    'individual_ratio_count': 0
                })
                df.loc[len(df)] = new_row
        # Sort by medal category to ensure consistent order
        df.sort_values('medal_category', inplace=True)
    
    # Plot bars for each group
    plt.bar(x - width*1.5, nyt_usa['individual_ratio_mean'], width, 
            label='NYT (USA Athletes)', color='#1f77b4')
    plt.bar(x - width/2, nyt_china['individual_ratio_mean'], width, 
            label='NYT (China Athletes)', color='#aec7e8')
    plt.bar(x + width/2, pd_usa['individual_ratio_mean'], width, 
            label='People\'s Daily (USA Athletes)', color='#ff9d5e')
    plt.bar(x + width*1.5, pd_china['individual_ratio_mean'], width, 
            label='People\'s Daily (China Athletes)', color='#ff7f0e')
    
    # Add labels and formatting
    plt.title('Individual Attribution by Medal Count and Nationality', fontsize=16)
    plt.ylabel('Individual Attribution Ratio\n(Higher = More Individual Focus)', fontsize=12)
    plt.xticks(x, medal_categories)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', alpha=0.3)
    
    # Set y-axis to range from 0 to 1
    plt.ylim(0, 1)
    
    # Add a horizontal line at 0.5 to indicate equal attribution
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'medal_count_attribution_comparison.png'), dpi=300)
    plt.close()

def plot_beijing_olympics_impact(beijing_impact):
    """
    Plot the impact of the 2008 Beijing Olympics on framing.
    
    Args:
        beijing_impact: Dictionary with Beijing Olympics impact statistics
    """
    plt.figure(figsize=(12, 7))
    
    # Extract data
    periods = ['pre_beijing', 'during_beijing', 'post_beijing']
    labels = ['2004-2007\n(Pre-Beijing)', '2008\n(Beijing Olympics)', '2009-2012\n(Post-Beijing)']
    
    nyt_values = [beijing_impact['NYT'][period]['mean'] for period in periods]
    nyt_std = [beijing_impact['NYT'][period]['std'] for period in periods]
    
    pd_values = [beijing_impact['PeoplesDaily'][period]['mean'] for period in periods]
    pd_std = [beijing_impact['PeoplesDaily'][period]['std'] for period in periods]
    
    # Set up positions for bars
    x = np.arange(len(periods))
    width = 0.35
    
    # Plot bars
    plt.bar(x - width/2, nyt_values, width, yerr=nyt_std, 
            label='NYT', color='#1f77b4', capsize=5)
    plt.bar(x + width/2, pd_values, width, yerr=pd_std, 
            label='People\'s Daily', color='#ff7f0e', capsize=5)
    
    # Add labels and formatting
    plt.title('Impact of 2008 Beijing Olympics on Individual vs. National Attribution', fontsize=16)
    plt.ylabel('Individual Attribution Ratio\n(Higher = More Individual Focus)', fontsize=12)
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # Set y-axis to range from 0 to 1
    plt.ylim(0, 1)
    
    # Add a horizontal line at 0.5 to indicate equal attribution
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    
    # Add value labels on bars
    for i, v in enumerate(nyt_values):
        plt.text(i - width/2, v + 0.02, f'{v:.2f}', ha='center')
    
    for i, v in enumerate(pd_values):
        plt.text(i + width/2, v + 0.02, f'{v:.2f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PATH, 'beijing_olympics_impact.png'), dpi=300)
    plt.close()

def generate_summary_statistics(nyt_framing, pd_framing):
    """
    Generate summary statistics for athlete framing analysis.
    
    Args:
        nyt_framing: DataFrame with NYT framing analysis results
        pd_framing: DataFrame with People's Daily framing analysis results
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'NYT': {
            'overall': {
                'mean_individual_ratio': nyt_framing['individual_ratio'].mean(),
                'median_individual_ratio': nyt_framing['individual_ratio'].median(),
                'std_individual_ratio': nyt_framing['individual_ratio'].std(),
                'count': len(nyt_framing),
                'individual_attribution_total': nyt_framing['individual_attribution_score'].sum(),
                'collective_attribution_total': nyt_framing['collective_attribution_score'].sum(),
                'overall_individual_ratio': (
                    nyt_framing['individual_attribution_score'].sum() / 
                    (nyt_framing['individual_attribution_score'].sum() + nyt_framing['collective_attribution_score'].sum())
                )
            }
        },
        'PeoplesDaily': {
            'overall': {
                'mean_individual_ratio': pd_framing['individual_ratio'].mean(),
                'median_individual_ratio': pd_framing['individual_ratio'].median(),
                'std_individual_ratio': pd_framing['individual_ratio'].std(),
                'count': len(pd_framing),
                'individual_attribution_total': pd_framing['individual_attribution_score'].sum(),
                'collective_attribution_total': pd_framing['collective_attribution_score'].sum(),
                'overall_individual_ratio': (
                    pd_framing['individual_attribution_score'].sum() / 
                    (pd_framing['individual_attribution_score'].sum() + pd_framing['collective_attribution_score'].sum())
                )
            }
        }
    }
    
    # Add statistics by nationality
    for nationality in ['USA', 'China']:
        nyt_nationality = nyt_framing[nyt_framing['nationality'] == nationality]
        pd_nationality = pd_framing[pd_framing['nationality'] == nationality]
        
        summary['NYT'][f'{nationality}_athletes'] = {
            'mean_individual_ratio': nyt_nationality['individual_ratio'].mean(),
            'median_individual_ratio': nyt_nationality['individual_ratio'].median(),
            'std_individual_ratio': nyt_nationality['individual_ratio'].std(),
            'count': len(nyt_nationality)
        }
        
        summary['PeoplesDaily'][f'{nationality}_athletes'] = {
            'mean_individual_ratio': pd_nationality['individual_ratio'].mean(),
            'median_individual_ratio': pd_nationality['individual_ratio'].median(),
            'std_individual_ratio': pd_nationality['individual_ratio'].std(),
            'count': len(pd_nationality)
        }
    
    # Add statistics for Olympic years vs. non-Olympic years
    nyt_olympic = nyt_framing[nyt_framing['article_year'].isin(olympic_years)]
    nyt_non_olympic = nyt_framing[~nyt_framing['article_year'].isin(olympic_years)]
    
    pd_olympic = pd_framing[pd_framing['article_year'].isin(olympic_years)]
    pd_non_olympic = pd_framing[~pd_framing['article_year'].isin(olympic_years)]
    
    summary['NYT']['olympic_years'] = {
        'mean_individual_ratio': nyt_olympic['individual_ratio'].mean(),
        'count': len(nyt_olympic)
    }
    
    summary['NYT']['non_olympic_years'] = {
        'mean_individual_ratio': nyt_non_olympic['individual_ratio'].mean(),
        'count': len(nyt_non_olympic)
    }
    
    summary['PeoplesDaily']['olympic_years'] = {
        'mean_individual_ratio': pd_olympic['individual_ratio'].mean(),
        'count': len(pd_olympic)
    }
    
    summary['PeoplesDaily']['non_olympic_years'] = {
        'mean_individual_ratio': pd_non_olympic['individual_ratio'].mean(),
        'count': len(pd_non_olympic)
    }
    
    return summary

def save_summary_statistics(summary, output_path):
    """
    Save summary statistics to a file.
    
    Args:
        summary: Dictionary with summary statistics
        output_path: Path to save the summary
    """
    # Save as pickle for programmatic access
    with open(os.path.join(output_path, 'athlete_framing_summary.pkl'), 'wb') as f:
        pickle.dump(summary, f)
    
    # Save as text for human readability
    with open(os.path.join(output_path, 'athlete_framing_summary.txt'), 'w') as f:
        f.write("Athlete Framing Analysis Summary Statistics\n")
        f.write("=========================================\n\n")
        
        for source in ['NYT', 'PeoplesDaily']:
            f.write(f"{source} Statistics:\n")
            f.write("-----------------\n")
            
            for category, stats in summary[source].items():
                f.write(f"  {category.replace('_', ' ').title()}:\n")
                for stat_name, stat_value in stats.items():
                    if isinstance(stat_value, float):
                        f.write(f"    {stat_name}: {stat_value:.4f}\n")
                    else:
                        f.write(f"    {stat_name}: {stat_value}\n")
                f.write("\n")
            
            f.write("\n")
    
    print(f"Summary statistics saved to {output_path}")

# -----------------------------------------------------------------------
# 5. Main Execution
# -----------------------------------------------------------------------

def main():
    """Main execution function."""
    start_time = time.time()
    
    # 1. Load datasets
    nyt_df, pd_df = load_datasets()
    usa_athletes, chinese_athletes = load_athlete_data()
    
    # 2. Extract athlete mentions (with time window consideration)
    nyt_mentions_path = os.path.join(OUTPUT_PATH, "nyt_athlete_mentions.pkl")
    pd_mentions_path = os.path.join(OUTPUT_PATH, "pd_athlete_mentions.pkl")
    
    if os.path.exists(nyt_mentions_path) and os.path.exists(pd_mentions_path):
        print("Loading previously extracted athlete mentions...")
        nyt_mentions = pd.read_pickle(nyt_mentions_path)
        pd_mentions = pd.read_pickle(pd_mentions_path)
    else:
        # Process articles to extract athlete mentions
        nyt_mentions = process_articles_parallel(nyt_df, {**usa_athletes, **chinese_athletes}, is_english=True)
        pd_mentions = process_articles_parallel(pd_df, {**usa_athletes, **chinese_athletes}, is_english=False)
        
        # Convert to DataFrames
        nyt_mentions = pd.DataFrame(nyt_mentions)
        pd_mentions = pd.DataFrame(pd_mentions)
        
        # Save mentions
        nyt_mentions.to_pickle(nyt_mentions_path)
        pd_mentions.to_pickle(pd_mentions_path)
    
    print(f"NYT athlete mentions: {len(nyt_mentions)}")
    print(f"People's Daily athlete mentions: {len(pd_mentions)}")
    
    # 3. Analyze individual vs. national framing
    nyt_framing_path = os.path.join(OUTPUT_PATH, "nyt_framing_analysis.pkl")
    pd_framing_path = os.path.join(OUTPUT_PATH, "pd_framing_analysis.pkl")
    
    if os.path.exists(nyt_framing_path) and os.path.exists(pd_framing_path):
        print("Loading previously analyzed framing data...")
        nyt_framing = pd.read_pickle(nyt_framing_path)
        pd_framing = pd.read_pickle(pd_framing_path)
    else:
        # Analyze framing
        nyt_framing = analyze_individual_vs_national_framing(nyt_mentions, is_english=True)
        pd_framing = analyze_individual_vs_national_framing(pd_mentions, is_english=False)
        
        # Save framing analysis
        nyt_framing.to_pickle(nyt_framing_path)
        pd_framing.to_pickle(pd_framing_path)
    
    # 4. Calculate aggregated statistics
    print("Calculating aggregated statistics...")
    
    # Yearly trends
    nyt_yearly = calculate_aggregate_statistics(nyt_framing, 'article_year')
    pd_yearly = calculate_aggregate_statistics(pd_framing, 'article_year')
    
    # Nationality bias
    nyt_nationality_bias = analyze_nationality_bias(nyt_framing, is_english=True)
    pd_nationality_bias = analyze_nationality_bias(pd_framing, is_english=False)
    
    # Medal count analysis
    nyt_medal_stats = analyze_athlete_prominence(nyt_framing)
    pd_medal_stats = analyze_athlete_prominence(pd_framing)
    
    # Beijing Olympics impact
    beijing_impact = analyze_beijing_olympics_impact(nyt_framing, pd_framing)
    
    # 5. Generate visualizations
    print("Generating visualizations...")
    
    # Plot individual vs. national attribution trends
    plot_individual_national_trends(nyt_yearly, pd_yearly)
    
    # Plot nationality comparison
    plot_nationality_comparison(nyt_nationality_bias, pd_nationality_bias)
    
    # Plot medal count comparison
    plot_medal_count_comparison(nyt_medal_stats, pd_medal_stats)
    
    # Plot Beijing Olympics impact
    plot_beijing_olympics_impact(beijing_impact)
    
    # 6. Generate and save summary statistics
    summary = generate_summary_statistics(nyt_framing, pd_framing)
    save_summary_statistics(summary, OUTPUT_PATH)
    
    # Print completion information
    total_time = time.time() - start_time
    print(f"\nAthlete framing analysis completed in {total_time:.2f} seconds")
    print(f"Results saved to {OUTPUT_PATH}")
    
    # Print key findings
    print("\nKey Findings:")
    print("-" * 40)
    
    nyt_individual_ratio = summary['NYT']['overall']['overall_individual_ratio']
    pd_individual_ratio = summary['PeoplesDaily']['overall']['overall_individual_ratio']
    
    print(f"Overall individual attribution ratio (NYT): {nyt_individual_ratio:.4f}")
    print(f"Overall individual attribution ratio (People's Daily): {pd_individual_ratio:.4f}")
    
    nyt_home_ratio = summary['NYT']['USA_athletes']['mean_individual_ratio']
    nyt_rival_ratio = summary['NYT']['China_athletes']['mean_individual_ratio']
    pd_home_ratio = summary['PeoplesDaily']['China_athletes']['mean_individual_ratio']
    pd_rival_ratio = summary['PeoplesDaily']['USA_athletes']['mean_individual_ratio']
    
    print(f"NYT attribution ratio for USA athletes: {nyt_home_ratio:.4f}")
    print(f"NYT attribution ratio for China athletes: {nyt_rival_ratio:.4f}")
    print(f"People's Daily attribution ratio for China athletes: {pd_home_ratio:.4f}")
    print(f"People's Daily attribution ratio for USA athletes: {pd_rival_ratio:.4f}")
    
    nyt_beijing = beijing_impact['NYT']['during_beijing']['mean']
    pd_beijing = beijing_impact['PeoplesDaily']['during_beijing']['mean']
    
    print(f"Individual attribution during Beijing Olympics (NYT): {nyt_beijing:.4f}")
    print(f"Individual attribution during Beijing Olympics (People's Daily): {pd_beijing:.4f}")
    
    print("-" * 40)

if __name__ == "__main__":
    main()