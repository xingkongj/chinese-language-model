import os
import re
import time
import json
import random
import logging
import argparse
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor

# 配置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# 用户代理列表，用于轮换请求头
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
]

# 新闻网站配置
NEWS_SITES = {
    "sina": {
        "url": "https://news.sina.com.cn/",
        "article_pattern": r"https?://[\w.]+sina\.com\.cn/[\w/]+/\d{4}-\d{2}-\d{2}/doc-[\w]+\.shtml",
        "title_selector": "h1.main-title",
        "content_selector": "div.article p",
    },
    "sohu": {
        "url": "https://www.sohu.com/",
        "article_pattern": r"https?://www\.sohu\.com/a/[\d]+_[\d]+",
        "title_selector": "h1.article-title",
        "content_selector": "article.article p",
    },
    "163": {
        "url": "https://www.163.com/",
        "article_pattern": r"https?://[\w.]+\.163\.com/[\w/]+/[\w/]+\.html",
        "title_selector": "h1.post_title",
        "content_selector": "div.post_body p",
    },
}

def get_random_user_agent():
    """获取随机用户代理
    
    Returns:
        str: 随机用户代理字符串
    """
    return random.choice(USER_AGENTS)

def fetch_url(url, max_retries=3, timeout=10):
    """获取URL内容
    
    Args:
        url: 要获取的URL
        max_retries: 最大重试次数
        timeout: 超时时间（秒）
        
    Returns:
        str: 页面内容
    """
    headers = {"User-Agent": get_random_user_agent()}
    retries = 0
    
    while retries < max_retries:
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            retries += 1
            logger.warning(f"获取URL失败: {url}, 错误: {e}, 重试: {retries}/{max_retries}")
            time.sleep(2 * retries)  # 指数退避
    
    logger.error(f"获取URL达到最大重试次数: {url}")
    return None

def extract_article_links(site_name, site_config, max_links=50):
    """从新闻网站提取文章链接
    
    Args:
        site_name: 网站名称
        site_config: 网站配置
        max_links: 最大链接数
        
    Returns:
        list: 文章链接列表
    """
    logger.info(f"从{site_name}提取文章链接")
    html_content = fetch_url(site_config["url"])
    if not html_content:
        return []
    
    article_pattern = site_config["article_pattern"]
    links = re.findall(article_pattern, html_content)
    unique_links = list(set(links))[:max_links]
    
    logger.info(f"从{site_name}提取了{len(unique_links)}个文章链接")
    return unique_links

def extract_article_content(url, site_config):
    """提取文章标题和正文
    
    Args:
        url: 文章URL
        site_config: 网站配置
        
    Returns:
        dict: 包含标题和正文的字典
    """
    html_content = fetch_url(url)
    if not html_content:
        return None
    
    soup = BeautifulSoup(html_content, "html.parser")
    
    # 提取标题
    title_element = soup.select_one(site_config["title_selector"])
    title = title_element.get_text().strip() if title_element else ""
    
    # 提取正文
    content_elements = soup.select(site_config["content_selector"])
    content = "\n".join([p.get_text().strip() for p in content_elements if p.get_text().strip()])
    
    if not title or not content:
        logger.warning(f"无法提取文章内容: {url}")
        return None
    
    return {
        "url": url,
        "title": title,
        "content": content,
    }

def collect_news_articles(site_name, site_config, max_articles=20, max_workers=5):
    """收集新闻文章
    
    Args:
        site_name: 网站名称
        site_config: 网站配置
        max_articles: 最大文章数
        max_workers: 最大线程数
        
    Returns:
        list: 文章列表
    """
    links = extract_article_links(site_name, site_config, max_links=max_articles*2)
    if not links:
        return []
    
    articles = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(extract_article_content, url, site_config): url for url in links[:max_articles]}
        for future in future_to_url:
            try:
                article = future.result()
                if article:
                    articles.append(article)
                    logger.info(f"成功提取文章: {article['title'][:30]}...")
            except Exception as e:
                url = future_to_url[future]
                logger.error(f"处理文章时出错: {url}, 错误: {e}")
    
    logger.info(f"从{site_name}成功收集了{len(articles)}篇文章")
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
        output_file = os.path.join(output_dir, f"news_articles_{int(time.time())}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        logger.info(f"已将{len(articles)}篇文章保存为JSON: {output_file}")
    else:
        output_file = os.path.join(output_dir, f"news_articles_{int(time.time())}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            for article in articles:
                f.write(f"{article['title']}\n\n")
                f.write(f"{article['content']}\n\n")
                f.write("="*50 + "\n\n")
        logger.info(f"已将{len(articles)}篇文章保存为TXT: {output_file}")
    
    return output_file

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="收集中文新闻数据")
    
    parser.add_argument("--sites", type=str, default="sina,sohu,163", help="要爬取的新闻网站，用逗号分隔")
    parser.add_argument("--articles_per_site", type=int, default=10, help="每个网站爬取的文章数量")
    parser.add_argument("--output_dir", type=str, default="../data/raw", help="输出目录")
    parser.add_argument("--format", type=str, choices=["txt", "json"], default="txt", help="输出格式")
    parser.add_argument("--max_workers", type=int, default=5, help="最大线程数")
    
    args = parser.parse_args()
    
    # 解析网站列表
    site_names = [s.strip() for s in args.sites.split(",") if s.strip() in NEWS_SITES]
    if not site_names:
        logger.error(f"未指定有效的新闻网站，可用网站: {', '.join(NEWS_SITES.keys())}")
        return
    
    # 收集文章
    all_articles = []
    for site_name in site_names:
        site_config = NEWS_SITES[site_name]
        articles = collect_news_articles(
            site_name, 
            site_config, 
            max_articles=args.articles_per_site,
            max_workers=args.max_workers
        )
        all_articles.extend(articles)
        # 添加延迟，避免请求过于频繁
        time.sleep(2)
    
    # 保存文章
    if all_articles:
        output_file = save_articles(all_articles, args.output_dir, args.format)
        logger.info(f"数据收集完成，共收集{len(all_articles)}篇文章")
    else:
        logger.warning("未收集到任何文章")

if __name__ == "__main__":
    main()