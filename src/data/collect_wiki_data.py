import os
import re
import time
import json
import logging
import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, unquote
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 维基百科API配置
WIKI_API_URL = "https://zh.wikipedia.org/w/api.php"
WIKI_ARTICLE_URL = "https://zh.wikipedia.org/wiki/"

def fetch_random_articles(num_articles=10):
    """获取随机维基百科文章
    
    Args:
        num_articles: 要获取的文章数量
        
    Returns:
        list: 文章标题列表
    """
    logger.info(f"获取{num_articles}篇随机维基百科文章")
    
    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnnamespace": "0",  # 主命名空间（文章）
        "rnlimit": str(num_articles),
    }
    
    try:
        response = requests.get(WIKI_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        articles = [article["title"] for article in data.get("query", {}).get("random", [])]
        logger.info(f"成功获取{len(articles)}篇随机文章标题")
        return articles
    except requests.RequestException as e:
        logger.error(f"获取随机文章失败: {e}")
        return []

def fetch_category_members(category, max_articles=50):
    """获取分类下的文章
    
    Args:
        category: 分类名称（不包含"Category:"前缀）
        max_articles: 最大文章数量
        
    Returns:
        list: 文章标题列表
    """
    logger.info(f"获取分类'{category}'下的文章")
    
    if not category.startswith("Category:") and not category.startswith("分类:"):
        category = f"Category:{category}"
    
    params = {
        "action": "query",
        "format": "json",
        "list": "categorymembers",
        "cmtitle": category,
        "cmlimit": str(min(max_articles, 500)),  # API限制
        "cmtype": "page",
    }
    
    try:
        response = requests.get(WIKI_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        articles = [article["title"] for article in data.get("query", {}).get("categorymembers", [])]
        logger.info(f"成功获取分类'{category}'下的{len(articles)}篇文章标题")
        return articles
    except requests.RequestException as e:
        logger.error(f"获取分类文章失败: {e}")
        return []

def search_articles(query, max_articles=50):
    """搜索维基百科文章
    
    Args:
        query: 搜索关键词
        max_articles: 最大文章数量
        
    Returns:
        list: 文章标题列表
    """
    logger.info(f"搜索关键词'{query}'相关的文章")
    
    params = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": query,
        "srlimit": str(min(max_articles, 500)),  # API限制
    }
    
    try:
        response = requests.get(WIKI_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        articles = [article["title"] for article in data.get("query", {}).get("search", [])]
        logger.info(f"成功搜索到{len(articles)}篇与'{query}'相关的文章标题")
        return articles
    except requests.RequestException as e:
        logger.error(f"搜索文章失败: {e}")
        return []

def fetch_article_content(title):
    """获取文章内容
    
    Args:
        title: 文章标题
        
    Returns:
        dict: 包含标题和内容的字典
    """
    logger.info(f"获取文章'{title}'的内容")
    
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": "true",  # 纯文本格式
        "exsectionformat": "plain",
    }
    
    try:
        response = requests.get(WIKI_API_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            logger.warning(f"未找到文章'{title}'")
            return None
        
        # 获取第一个（也是唯一的）页面
        page_id = list(pages.keys())[0]
        page = pages[page_id]
        
        # 检查是否有内容
        if "extract" not in page or not page["extract"].strip():
            logger.warning(f"文章'{title}'没有内容")
            return None
        
        article_url = urljoin(WIKI_ARTICLE_URL, title.replace(" ", "_"))
        
        return {
            "title": page.get("title", title),
            "content": page.get("extract", ""),
            "url": article_url,
        }
    except requests.RequestException as e:
        logger.error(f"获取文章'{title}'内容失败: {e}")
        return None
    except Exception as e:
        logger.error(f"处理文章'{title}'时出错: {e}")
        return None

def clean_wiki_text(text):
    """清理维基百科文本
    
    Args:
        text: 原始文本
        
    Returns:
        str: 清理后的文本
    """
    if not text:
        return ""
    
    # 移除引用标记 [1], [2], 等
    text = re.sub(r'\[\d+\]', '', text)
    
    # 移除多余的空白行
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def collect_wiki_articles(titles, max_workers=5):
    """收集维基百科文章
    
    Args:
        titles: 文章标题列表
        max_workers: 最大线程数
        
    Returns:
        list: 文章列表
    """
    if not titles:
        return []
    
    articles = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_title = {executor.submit(fetch_article_content, title): title for title in titles}
        for future in future_to_title:
            try:
                article = future.result()
                if article and article["content"]:
                    # 清理文本
                    article["content"] = clean_wiki_text(article["content"])
                    if article["content"]:
                        articles.append(article)
                        logger.info(f"成功获取文章: {article['title']}")
            except Exception as e:
                title = future_to_title[future]
                logger.error(f"处理文章'{title}'时出错: {e}")
    
    logger.info(f"成功收集了{len(articles)}篇维基百科文章")
    return articles

def save_articles(articles, output_dir, format="txt"):
    """保存文章到文件
    
    Args:
        articles: 文章列表
        output_dir: 输出目录
        format: 输出格式 (txt 或 json)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if format.lower() == "json":
        output_file = os.path.join(output_dir, f"wiki_articles_{int(time.time())}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        logger.info(f"已将{len(articles)}篇文章保存为JSON: {output_file}")
    else:
        output_file = os.path.join(output_dir, f"wiki_articles_{int(time.time())}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            for article in articles:
                f.write(f"{article['title']}\n\n")
                f.write(f"{article['content']}\n\n")
                f.write("="*50 + "\n\n")
        logger.info(f"已将{len(articles)}篇文章保存为TXT: {output_file}")
    
    return output_file

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="收集中文维基百科数据")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--random", type=int, help="获取指定数量的随机文章")
    group.add_argument("--category", type=str, help="获取指定分类下的文章")
    group.add_argument("--search", type=str, help="搜索指定关键词的文章")
    
    parser.add_argument("--max_articles", type=int, default=50, help="最大文章数量")
    parser.add_argument("--output_dir", type=str, default="../data/raw", help="输出目录")
    parser.add_argument("--format", type=str, choices=["txt", "json"], default="txt", help="输出格式")
    parser.add_argument("--max_workers", type=int, default=5, help="最大线程数")
    
    args = parser.parse_args()
    
    # 获取文章标题
    titles = []
    if args.random:
        titles = fetch_random_articles(args.random)
    elif args.category:
        titles = fetch_category_members(args.category, args.max_articles)
    elif args.search:
        titles = search_articles(args.search, args.max_articles)
    
    if not titles:
        logger.error("未获取到任何文章标题")
        return
    
    # 收集文章内容
    articles = collect_wiki_articles(titles, args.max_workers)
    
    # 保存文章
    if articles:
        output_file = save_articles(articles, args.output_dir, args.format)
        logger.info(f"数据收集完成，共收集{len(articles)}篇文章")
    else:
        logger.warning("未收集到任何文章内容")

if __name__ == "__main__":
    main()