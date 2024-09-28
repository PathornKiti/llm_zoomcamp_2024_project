from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urlparse
import re
import asyncio
import pandas as pd
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch, NotFoundError

# Elasticsearch client setup
es_client = Elasticsearch('http://elasticsearch:9200')

# Index settings
index_name = "relationship_consult"
index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "source": {"type": "text"},
            "title": {"type": "text"},
            "description": {"type": "text"},
            "section": {"type": "keyword"},
            "content": {"type": "text"},
            "content_vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            },
            "title_description_vector": {
                "type": "dense_vector",
                "dims": 768,
                "index": True,
                "similarity": "cosine"
            },
        }
    }
}

# Function to load URLs asynchronously using AsyncChromiumLoader
async def get_blog_urls(base_url: str):
    loader = AsyncChromiumLoader([base_url])
    docs = await loader.aload()
    html_content = docs[0].page_content
    return html_content

# Function to clean up the content
def clean_content(content):
    content = content.replace('\n', '').replace('\xa0', ' ')
    content = re.sub(r'\s+', ' ', content).strip()
    content = re.sub(r'Post Views:.*', '', content, flags=re.DOTALL)
    return content

# Find last page for pagination
async def find_last_page(base_url: str):
    html_content = await get_blog_urls(base_url)
    soup = BeautifulSoup(html_content, 'html.parser')
    pagination = soup.find('div', class_='box-pagination')
    if pagination:
        pages = pagination.find_all('a', class_='page-numbers')
        last_page = max([int(page.text) for page in pages if page.text.isdigit()])
        print(f"The last page number is: {last_page}")
        return last_page
    else:
        print("Pagination not found.")
        return 1

# Find all blog URLs for a given category
async def find_blog_urls(base_url_template, last_page):
    blog_urls = []
    for i in tqdm(range(1, last_page + 1)):
        base_url = base_url_template.format(i)
        html_content = await get_blog_urls(base_url)
        soup = BeautifulSoup(html_content, 'html.parser')
        hrefs = [a['href'] for a in soup.find_all('a', href=True)]
        filtered_urls = [href for href in hrefs if href.startswith('https://www.alljitblog.com/')]
        filtered_urls = filtered_urls[1:]
        filtered_urls = list(set(filtered_urls))
        unwanted_substrings = ['?cat', '.com/#', '/author/admin-alljit/', '/category/']
        filtered_urls = [url for url in filtered_urls if not any(substring in url for substring in unwanted_substrings)]
        blog_urls.extend(filtered_urls)

    blog_urls = list({urlparse(url).scheme + '://' + urlparse(url).netloc + urlparse(url).path for url in blog_urls})
    if 'https://www.alljitblog.com/' in blog_urls:
        blog_urls.remove('https://www.alljitblog.com/')
    
    return blog_urls

# Load content from blog URLs and clean
def load_blog_content(blog_urls):
    loader_multiple_pages = WebBaseLoader(blog_urls, encoding='utf-8')
    blog_data = loader_multiple_pages.load()
    
    for data in blog_data:
        data.page_content = data.page_content.replace('\n', '')
        data.page_content = data.page_content.replace('\xa0', ' ')
        data.page_content = re.sub(r'\s+', ' ', data.page_content).strip()
        data.page_content = clean_content(data.page_content)
    
    return blog_data

# Check if the Elasticsearch client is available
def check_elasticsearch_client():
    try:
        if es_client.ping():
            print("Elasticsearch is available.")
        else:
            print("Elasticsearch is unavailable.")
    except Exception as e:
        print(f"Error connecting to Elasticsearch: {e}")

# Create Elasticsearch index if it doesn't already exist
def create_index():
    try:
        if not es_client.indices.exists(index=index_name):
            es_client.indices.create(index=index_name, body=index_settings)
            print(f"Index '{index_name}' created.")
        else:
            print(f"Index '{index_name}' already exists.")
    except Exception as e:
        print(f"Error creating index: {e}")

# Check if the document exists in Elasticsearch by title
def document_exists(title):
    query = {
        "query": {
            "match": {
                "title": title
            }
        }
    }
    try:
        result = es_client.search(index=index_name, body=query)
        return result['hits']['total']['value'] > 0
    except NotFoundError:
        return False

# Main function to orchestrate the tasks
async def main():
    # Model setup for embedding
    model_name = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    model = SentenceTransformer(model_name)
    
    # Step 1: Process 'จิตวิทยาชีวิตคู่' category
    base_url_dating = "https://www.alljitblog.com/category/จิตวิทยาชีวิตคู่/"
    last_page_dating = await find_last_page(base_url_dating)
    dating_blog_urls = await find_blog_urls("https://www.alljitblog.com/category/จิตวิทยาชีวิตคู่/page/{}/", last_page_dating)
    dating_data = load_blog_content(dating_blog_urls)
    
    # Step 2: Process 'psychiatrist' category
    base_url_psy = "https://www.alljitblog.com/category/psychiatrist/"
    last_page_psy = await find_last_page(base_url_psy)
    psy_blog_urls = await find_blog_urls("https://www.alljitblog.com/category/psychiatrist/page/{}/", last_page_psy)
    psy_data = load_blog_content(psy_blog_urls)
    
    # Convert scraped data into a DataFrame
    relationship_df = pd.DataFrame([{'source': d.metadata['source'], 'title': d.metadata.get('title'), 'description': d.metadata.get('description', ''), 'content': d.page_content, 'section': 'relationship'} for d in dating_data])
    psy_df = pd.DataFrame([{'source': d.metadata['source'], 'title': d.metadata.get('title'), 'description': d.metadata.get('description', ''), 'content': d.page_content, 'section': 'psycology'} for d in psy_data])
    
    all_df = pd.concat([relationship_df, psy_df], ignore_index=True)
    all_df = all_df.drop_duplicates(subset=['title'])
    
    # Prepare documents for Elasticsearch
    documents = []
    for _, row in all_df.iterrows():
        documents.append({
            "source": row['source'], 
            "title": row["title"],
            "description": row["description"],
            "content": row["content"],
            "section": row["section"]
        })
    
    # Generate embeddings and index documents
    for doc in tqdm(documents):
        title = doc['title']
        
        # Check if the document already exists in Elasticsearch
        if not document_exists(title):
            content = doc['content']
            description = doc['description']
            td = f"Title:{title}\nDescription:{description}"
            
            # Generate vectors
            doc['content_vector'] = model.encode(content)
            doc['title_description_vector'] = model.encode(td)
            
            # Index the document in Elasticsearch
            es_client.index(index=index_name, document=doc)
            print(f"Indexed document: {title}")
        else:
            print(f"Document '{title}' already exists. Skipping.")

if __name__ == "__main__":
    check_elasticsearch_client()
    create_index()
    asyncio.run(main())
