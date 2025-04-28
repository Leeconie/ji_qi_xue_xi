import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

# 下载必要的NLTK资源（首次运行需要）
# nltk.download('punkt')
# nltk.download('stopwords')

# 加载spaCy中文模型（首次使用需要安装）
# 安装命令: pip install spacy
# 下载中文模型: python -m spacy download zh_core_web_sm
try:
    nlp = spacy.load('zh_core_web_sm')
    print("成功加载中文spaCy模型")
except:
    print("请先安装spaCy中文模型: python -m spacy download zh_core_web_sm")
    # 如果中文模型不可用，尝试使用英文模型
    try:
        nlp = spacy.load('en_core_web_sm')
        print("使用英文spaCy模型作为替代")
    except:
        print("请先安装spaCy英文模型: python -m spacy download en_core_web_sm")

# 读取CSV文件
data_path = "tweets.csv"  # 请替换为您实际的文件路径
try:
    df = pd.read_csv(data_path)
    print("文件读取成功！")
    print(f"数据形状: {df.shape}")
    print("\n前5行数据:")
    print(df.head())
except Exception as e:
    print(f"读取文件时出错: {e}")
    
# 假设我们要从推文中提取用户名、标签和URL
def extract_info(text):
    # 提取用户名 (@username)
    usernames = re.findall(r'@([A-Za-z0-9_]+)', str(text))
    
    # 提取标签 (#hashtag)
    hashtags = re.findall(r'#([A-Za-z0-9_]+)', str(text))
    
    # 提取URL
    urls = re.findall(r'https?://[^\s]+', str(text))
    
    return {
        'usernames': usernames,
        'hashtags': hashtags,
        'urls': urls
    }

# 命名实体识别函数
def extract_entities(text):
    doc = nlp(str(text))
    entities = {}
    
    # 提取所有实体
    for ent in doc.ents:
        entity_type = ent.label_
        entity_text = ent.text
        
        if entity_type not in entities:
            entities[entity_type] = []
        
        entities[entity_type].append(entity_text)
    
    return entities

# 假设文本内容在'text'列中
if 'text' in df.columns:
    # 应用提取函数到每一行
    extracted_info = df['text'].apply(extract_info)
    
    # 将提取的信息添加为新列
    df['extracted_usernames'] = extracted_info.apply(lambda x: x['usernames'])
    df['extracted_hashtags'] = extracted_info.apply(lambda x: x['hashtags'])
    df['extracted_urls'] = extracted_info.apply(lambda x: x['urls'])
    
    print("\n信息提取后的数据:")
    print(df[['text', 'extracted_usernames', 'extracted_hashtags', 'extracted_urls']].head())
    
    # 应用命名实体识别
    print("\n正在进行命名实体识别，这可能需要一些时间...")
    df['entities'] = df['text'].apply(extract_entities)
    
    # 展示实体识别结果
    print("\n命名实体识别结果:")
    for i, row in df.head().iterrows():
        print(f"\n文本 {i+1}: {row['text'][:100]}...")
        if row['entities']:
            for entity_type, entities in row['entities'].items():
                print(f"  - {entity_type}: {', '.join(entities)}")
        else:
            print("  未识别到实体")
else:
    print("数据中没有'text'列，请检查您的数据结构并修改代码中的列名")

# 基本统计
if 'text' in df.columns:
    # 计算每条推文的字数
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))
    
    # 计算每条推文的字符数
    df['char_count'] = df['text'].apply(lambda x: len(str(x)))
    
    print("\n文本统计信息:")
    print(df[['text', 'word_count', 'char_count']].head())

# 实体统计与可视化
if 'entities' in df.columns:
    # 统计所有实体类型及其出现次数
    entity_types = []
    for entities_dict in df['entities']:
        entity_types.extend(entities_dict.keys())
    
    entity_type_counts = Counter(entity_types)
    
    print("\n实体类型统计:")
    for entity_type, count in entity_type_counts.most_common():
        print(f"{entity_type}: {count}次")
    
    # 可视化实体类型分布
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(entity_type_counts.keys()), y=list(entity_type_counts.values()))
    plt.title('命名实体类型分布')
    plt.xlabel('实体类型')
    plt.ylabel('出现次数')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('entity_type_distribution.png')
    print("\n已保存实体类型分布图表到 'entity_type_distribution.png'")
    
    # 提取最常见的实体
    all_entities = []
    for entities_dict in df['entities']:
        for entity_type, entities in entities_dict.items():
            all_entities.extend(entities)
    
    common_entities = Counter(all_entities).most_common(20)
    
    print("\n最常见的20个实体:")
    for entity, count in common_entities:
        print(f"{entity}: {count}次")
    
    # 可视化最常见实体
    plt.figure(figsize=(12, 8))
    entity_names = [e[0] for e in common_entities]
    entity_counts = [e[1] for e in common_entities]
    sns.barplot(x=entity_counts, y=entity_names)
    plt.title('最常见的20个命名实体')
    plt.xlabel('出现次数')
    plt.ylabel('实体')
    plt.tight_layout()
    plt.savefig('common_entities.png')
    print("\n已保存常见实体图表到 'common_entities.png'")

# 保存处理后的数据
try:
    # 将实体字典转换为字符串以便保存
    df['entities_str'] = df['entities'].apply(lambda x: str(x))
    df.to_csv('tweets_with_entities.csv', index=False)
    print("\n已将处理后的数据保存到 'tweets_with_entities.csv'")
except Exception as e:
    print(f"\n保存数据时出错: {e}")