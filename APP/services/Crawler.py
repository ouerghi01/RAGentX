from services.notes import exs,tabs
import requests
import re
import threading
import string
from langchain_core.documents import Document

from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig,RateLimiter,CrawlerMonitor, DisplayMode
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
def generate_exercise_url(ex):
    base_url=f"https://github.com/echiner/edem-mda-nosql-cassandra/blob/main/Exercises/{ex}/README.md"
    return base_url
def generate_basic_link(e):
        basic_link=f"https://www.tutorialspoint.com/cassandra/cassandra_{e}.htm"
        return basic_link
def get_links_from_cassandra_apache():
    url_scrap="https://cassandra.apache.org/sitemap.xml"
    url_base="https://cassandra.apache.org"
    response=requests.get(url_scrap)
    links=re.findall(r"<loc>(.*?)</loc>",response.text)
    length_links=len(links)
    orginal_links=[]
    for link_miss in links:
         link=re.split(r'/_/',link_miss)[0]
         if link !="":
            orginal_links.append(url_base+link)
    length_orginal_links=len(orginal_links)
    if length_orginal_links < length_links:
        print("Some links were missed")
    return orginal_links
         
def clean_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = BeautifulSoup(text, "html.parser").get_text()  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    
    return text
def process_markdown_result(result,contents):
    text = clean_text(result.markdown)
    contents.append(text)

async def main_crawler():
    
    
    urls=[generate_exercise_url(ex) for ex in exs]
    urls_new=[
        generate_basic_link(e) for e in tabs
    ]
    all_urls=urls+urls_new +get_links_from_cassandra_apache()
    all_urls.append("https://www.freecodecamp.org/news/the-apache-cassandra-begin")
    all_urls.append("https://stackoverflow.com/questions/47148035/datastax-cassandra-seems-expensive-is-there-a-best-practice-configuration-to-us")
    all_urls.append("https://github.com/DataStax-Academy/workshop-cassandra-certification/blob/main/PRACTICE.md")
    all_urls.append("https://github.com/datastax/cql-proxy")
    browser_config = BrowserConfig(headless=True, verbose=False)
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        stream=False  # Default: get all results at once
    )
    dispatcher = MemoryAdaptiveDispatcher(
    memory_threshold_percent=90.0,  # Pause if memory exceeds this
    check_interval=1.0,             # How often to check memory
    max_session_permit=10,          # Maximum concurrent tasks
    rate_limiter=RateLimiter(       # Optional rate limiting
        base_delay=(1.0, 2.0),
        max_delay=30.0,
        max_retries=2
    ),
    monitor=CrawlerMonitor(         # Optional monitoring
        max_visible_rows=15,
        display_mode=DisplayMode.DETAILED
    )
    )   
    contents=[]
    async with AsyncWebCrawler(config=browser_config) as crawler:
            results = await crawler.arun_many(
            urls=all_urls,
            config=run_config,
            dispatcher=dispatcher
            )
            threads=[]
            for result in results:
                thread_process=threading.Thread(target=process_markdown_result,args=(result,contents))
                threads.append(thread_process)
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            docs=[]
            for i,content in enumerate(contents):
                 doc=Document(page_content=content,metadata={
                      "meta":all_urls[i]
                 })
                 docs.append(doc)
            return docs

