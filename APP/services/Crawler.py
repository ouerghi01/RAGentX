
import re
import threading
import string

from bs4 import BeautifulSoup
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig,RateLimiter,CrawlerMonitor, DisplayMode
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher


         
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

async def crawler_Info(all_urls):
    
    
    
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
                 
                 docs.append(content)
            return "\n".join(docs)

