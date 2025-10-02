# ingestion.py
import requests
from bs4 import BeautifulSoup, Comment
import re
import nltk
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss # For vector store
import os
from dotenv import load_dotenv
import time # For polite scraping

load_dotenv()

# Download punkt tokenizer for sentence splitting if not already present
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "france_land_sections.index")
CHUNKS_DATA_PATH = os.path.join(DATA_DIR, "france_land_sections_chunks_data.json")

# List of URLs to process
FRANCE_LAND_URLS = [
    "https://www.britannica.com/place/France/Land",
    "https://www.britannica.com/place/France/The-Hercynian-massifs",
    "https://www.britannica.com/place/France/The-great-lowlands",
    "https://www.britannica.com/place/France/The-younger-mountains-and-adjacent-plains",
    "https://www.britannica.com/place/France/Drainage",
    "https://www.britannica.com/place/France/Soils",
    "https://www.britannica.com/place/France/Climate",
    "https://www.britannica.com/place/France/Plant-and-animal-life"
]

def fetch_article_content(url: str):
    """Fetches and extracts the main content and title from a Britannica URL."""
    try:
        print(f"Fetching: {url}")
        # Add a small delay to be polite to the server
        time.sleep(1) 
        response = requests.get(url, timeout=15, headers={'User-Agent': 'RAGIngestionBot/1.0'})
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None

def extract_text_and_metadata_britannica(soup: BeautifulSoup, url: str):
    """
    Extracts title, main text content, and other metadata from the parsed HTML,
    focusing on sections within div.topic-content.
    """
    if not soup:
        return None, {}

    # Overall Page Title (usually the main H1 of the page)
    page_title_tag = soup.find('h1', class_=lambda x: x != 'h3') # Try to get the main H1
    if not page_title_tag: # Fallback if specific class logic is complex
        page_title_tag = soup.find('h1')
    page_title = page_title_tag.get_text(strip=True) if page_title_tag else "Unknown Page Title"


    # Metadata: Last Updated Date
    date_tag = soup.find('span', class_='last-updated')
    date_str = date_tag.get_text(strip=True).replace("Last Updated: ", "") if date_tag else "Unknown Date"

    # Metadata: Category (from breadcrumbs)
    breadcrumbs = soup.select('nav[aria-label="breadcrumb"] ol li a')
    category = "Geography" # Default for these pages
    if breadcrumbs and len(breadcrumbs) > 1:
        relevant_crumbs = [b.get_text(strip=True) for b in breadcrumbs if b.get_text(strip=True).lower() not in ["home", page_title.lower()]]
        if relevant_crumbs:
            # Take the one before "France" or a suitable higher-level category
            if "France" in relevant_crumbs and relevant_crumbs.index("France") > 0:
                category = relevant_crumbs[relevant_crumbs.index("France")-1]
            elif len(relevant_crumbs) > 1:
                 category = relevant_crumbs[-2] # Often a good bet
            else:
                category = relevant_crumbs[-1]


    # Main Content Extraction
    content_texts = []
    
    # The primary content area for Britannica articles
    topic_content_div = soup.find('div', class_='topic-content')
    if not topic_content_div:
        # Fallback for pages like "/Land" which might have a slightly different top-level structure
        # but content is still often within section tags with IDs
        topic_content_div = soup.find('div', id='content') # Another common main content wrapper
        if not topic_content_div:
            print(f"Warning: Could not find 'div.topic-content' or 'div#content' in {url}. Attempting broader search.")
            # If still not found, search for sections directly under body or main
            # This is less precise and might pick up unwanted sections
            candidate_containers = soup.find_all(['main', 'article', 'div.reading-channel'])
            if not candidate_containers: candidate_containers = [soup.body] # Absolute fallback
            
            # Search for sections with 'ref' in their id, a common Britannica pattern
            all_sections = []
            for container in candidate_containers:
                if container:
                    all_sections.extend(container.find_all('section', id=lambda x: x and 'ref' in x))
            
            if not all_sections: # If no sections with 'ref' IDs, try any section
                 for container in candidate_containers:
                    if container:
                        all_sections.extend(container.find_all('section'))
            
            # If still no sections, look for article tag
            if not all_sections:
                article_tag = soup.find('article')
                if article_tag:
                    all_sections = [article_tag] # Treat the whole article tag as one section

        else: # Found 'div#content'
             all_sections = topic_content_div.find_all('section', id=lambda x: x and 'ref' in x)
             if not all_sections: # If no 'ref' sections, take all direct section children
                 all_sections = topic_content_div.find_all('section', recursive=False)


    else: # Found 'div.topic-content'
        # Find all <section> tags that are likely main content blocks (often have an 'id' attribute)
        # Or, specifically target sections that seem to contain the text.
        # The comments <!--[PREMOD1]--> etc. are inside these sections or around paragraphs.
        
        # Strategy: iterate through direct children of topic_content_div.
        # If child is a section with 'ref' id, process it.
        # Also look for content between known comment markers if sections are not clearly defined.
        
        # More robust: find all 'section' tags with an 'id' starting with 'ref' within 'topic-content'
        all_sections = topic_content_div.find_all('section', id=lambda x: x and 'ref' in x)

        if not all_sections: # If no sections with 'ref' are found
            # This can happen if the content is not in such sections, e.g. the main "/Land" page introduction
            # In this case, try to get all p.topic-paragraph directly under topic_content_div
            # or grouped by h2s
            headings = topic_content_div.find_all(['h2', 'h3'], class_=lambda x: x and ('h1','h2','h3','h4','h5','h6') in x.split() or not x) # Find H2s or H3s that act as headings
            
            current_elements = []
            if not headings: # No subheadings, just take all paragraphs
                for p_tag in topic_content_div.find_all('p', class_='topic-paragraph'):
                    current_elements.append(p_tag.get_text(separator=' ', strip=True))
                if current_elements:
                     content_texts.append("\n".join(current_elements))
            else:
                # Group paragraphs under their preceding headings
                for el in topic_content_div.children:
                    if el.name in ['h2','h3'] and (not el.get('class') or any(h_class in el.get('class',[]) for h_class in ['h1','h2','h3','h4','h5','h6'])):
                        if current_elements: # Add previous section's text
                            content_texts.append("\n".join(current_elements))
                            current_elements = []
                        current_elements.append(el.get_text(separator=' ', strip=True)) # Add heading text
                    elif el.name == 'p' and el.has_attr('class') and 'topic-paragraph' in el['class']:
                        current_elements.append(el.get_text(separator=' ', strip=True))
                    elif el.name == 'div' and el.find_all('p', class_='topic-paragraph'): # handle paragraphs inside divs
                         for p_in_div in el.find_all('p', class_='topic-paragraph'):
                            current_elements.append(p_in_div.get_text(separator=' ', strip=True))

                if current_elements: # Add the last section's text
                    content_texts.append("\n".join(current_elements))
    
    # Process identified sections
    for section in all_sections:
        section_text_parts = []
        # Extract heading from the section (h2, or h1/h3 that functions as a section title)
        heading_tag = section.find(['h1', 'h2', 'h3']) # More general
        if heading_tag:
            section_text_parts.append(heading_tag.get_text(strip=True))

        # Extract all relevant paragraphs within this section
        paragraphs = section.find_all('p', class_='topic-paragraph')
        if not paragraphs: # Fallback if no 'topic-paragraph' class, get all 'p'
            paragraphs = section.find_all('p')
            
        for p_tag in paragraphs:
            # Remove content of 'span' tags with class 'md-ref' (citation links)
            for md_ref_span in p_tag.find_all('span', class_='md-ref'):
                md_ref_span.decompose()
            section_text_parts.append(p_tag.get_text(separator=' ', strip=True))
        
        if section_text_parts:
            content_texts.append("\n".join(section_text_parts))

    # Remove comments from the soup before final text extraction if relying on broad text grab
    # This is more for a general extraction from a less structured part
    if not content_texts and topic_content_div: # If specific section parsing failed, try to get all text from topic_content_div
        print(f"Falling back to broader text extraction for {url} within 'topic-content' or equivalent.")
        comments = topic_content_div.find_all(string=lambda text: isinstance(text, Comment))
        for comment in comments:
            comment.extract()
        
        # General text grab from the topic_content_div after comment removal
        # Get text from P, H2, H3 elements directly under topic_content_div or nested
        elements_for_text = topic_content_div.find_all(['p', 'h2', 'h3', 'h4', 'li']) # Include list items
        temp_content = []
        for el in elements_for_text:
            # Avoid known noisy elements if any by checking class or id
            if el.has_attr('class') and ('related-topic' in el['class'] or 'reference' in el['class']):
                continue
            text_ = el.get_text(separator=' ', strip=True)
            if len(text_.split()) > 3 : # Only add if it has some substance
                temp_content.append(text_)
        content_texts.append("\n".join(temp_content))


    final_text = "\n\n".join(filter(None, content_texts)) # Join texts from all sections/parts

    # If by any chance the title of the current subsection page is more specific, use it.
    # e.g. for "France/The-Hercynian-massifs", the page_title might be "The Hercynian massifs"
    # and the overall article title might be "France". We want the most specific one.
    # The current `page_title` should be specific enough from the `h1` tag.

    metadata = {
        "source_url": url,
        "title": page_title, # This should be the specific title of the current page/section
        "publication_date": date_str,
        "category": category,
    }
    
    return final_text, metadata

# B. Cleaning (same as before)
def clean_text(text: str) -> str:
    if not text: return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'<[^>]+>', '', text) 
    text = re.sub(r'\[\d+\]', '', text) # Remove [1], [2]
    text = re.sub(r'\s*advertisement\s*', ' ', text, flags=re.IGNORECASE) # Remove "advertisement"
    # Remove text like "The editors of Encyclopaedia Britannica"
    text = re.sub(r'the editors of encyclopaedia britannica(?:\s*this article was most recently revised and updated by.*?)?(?:\s*see article history)?', '', text, flags=re.IGNORECASE).strip()
    # Remove "Read More on This Topic" and similar phrases often found at end of sections
    text = re.sub(r'read more on this topic\s*:\s*[\w\s,]+', '', text, flags=re.IGNORECASE).strip()
    # Normalize consecutive newlines that might have become spaces then re-strip
    text = re.sub(r'\s*\n\s*', '\n', text).strip()
    text = re.sub(r' +', ' ', text) # Consolidate multiple spaces
    return text

# C. Chunking (same as before, but justification might be reiterated or refined if needed)
def chunk_text_sentences(text: str, sentences_per_chunk: int = 4, overlap_sentences: int = 1) -> list[str]:
    """
    Splits text into sentences and then groups them into overlapping chunks.
    Justification:
    - Method: Sentence splitting respects natural language boundaries.
    - Size (sentences_per_chunk=4): Aims for chunks ~60-160 words, good for models like all-MiniLM-L6-v2.
    - Overlap (overlap_sentences=1): Ensures context continuity.
    """
    if not text: return []
    sentences = nltk.sent_tokenize(text)
    if not sentences: return []
    
    chunks = []
    i = 0
    while i < len(sentences):
        end = i + sentences_per_chunk
        chunk_sentences = sentences[i:end]
        chunks.append(" ".join(chunk_sentences))
        
        next_i = i + sentences_per_chunk - overlap_sentences
        if next_i <= i : # Ensure progression if overlap is large or sentences_per_chunk is small
            next_i = i + 1 
        i = next_i
        
        # Avoid creating a very small last chunk that is mostly overlap of the previous one
        if i >= len(sentences) - overlap_sentences and len(sentences) > sentences_per_chunk:
            # If the remaining part is only the overlap of what would be a full next chunk, don't add.
            # However, if there's substantial new content, add it.
            remaining_count = len(sentences) - (i - (sentences_per_chunk - overlap_sentences))
            if remaining_count <= overlap_sentences and overlap_sentences > 0 :
                 # Check if the last added chunk already covers the end
                 last_chunk_end_sentence_idx = (i - (sentences_per_chunk - overlap_sentences)) + sentences_per_chunk -1
                 if last_chunk_end_sentence_idx >= len(sentences) -1:
                    break # Already covered
            
    # Post-process chunks: ensure they are not too short or empty
    return [chunk for chunk in chunks if len(chunk.split()) > 10] # Min 10 words for a meaningful chunk

# D. Embedding Model Choice & Implementation (same as before)
def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print("Embedding model loaded.")
    return model

def create_embeddings(chunks: list[str], model):
    if not chunks: return np.array([])
    print(f"Generating embeddings for {len(chunks)} chunks...")
    embeddings = model.encode(chunks, show_progress_bar=True)
    print("Embeddings generated.")
    return embeddings

def build_and_save_faiss_index(embeddings: np.ndarray, chunk_data: list[dict]):
    if embeddings.shape[0] == 0:
        print("No embeddings to index.")
        return

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))

    print(f"Saving FAISS index to {FAISS_INDEX_PATH}")
    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"Saving chunk data to {CHUNKS_DATA_PATH}")
    with open(CHUNKS_DATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunk_data, f, indent=4)

def run_ingestion_pipeline():
    """Main function to run the entire ingestion pipeline for multiple URLs."""
    
    all_raw_texts_with_metadata = [] # Store tuples of (text, metadata)

    for url in FRANCE_LAND_URLS:
        soup = fetch_article_content(url)
        if not soup:
            print(f"Skipping {url} due to fetch error.")
            continue

        raw_text, metadata = extract_text_and_metadata_britannica(soup, url)
        if not raw_text:
            print(f"No text content extracted from {url}. Skipping.")
            continue
        
        print(f"Successfully extracted from {url}: Title: {metadata.get('title')}, Text length: {len(raw_text)}")
        all_raw_texts_with_metadata.append({"text": raw_text, "metadata": metadata})

    if not all_raw_texts_with_metadata:
        print("No data extracted from any URL. Exiting.")
        return

    # Process all collected texts
    all_chunks_with_metadata = []
    
    for item in all_raw_texts_with_metadata:
        cleaned_text = clean_text(item["text"])
        if not cleaned_text:
            print(f"Skipping {item['metadata']['source_url']} after cleaning, no text remained.")
            continue
            
        chunks = chunk_text_sentences(cleaned_text) # Default: 4 sentences/chunk, 1 overlap
        
        for i, chunk_text in enumerate(chunks):
            # Each chunk inherits metadata from its source page
            # and gets a unique chunk_id within that page context (or globally)
            chunk_specific_metadata = item["metadata"].copy()
            # Create a more robust global chunk ID if needed, e.g. based on URL and index
            # For simplicity, using a running counter for the demo ID in processed_chunks_data later.
            
            all_chunks_with_metadata.append({
                "text": chunk_text,
                "metadata": chunk_specific_metadata
                # chunk_id will be added when preparing final list for FAISS
            })
            
    if not all_chunks_with_metadata:
        print("No chunks generated from any processed text. Exiting.")
        return
        
    print(f"Total chunks generated from all pages: {len(all_chunks_with_metadata)}")

    # Prepare data for embedding and FAISS
    # We need a list of just the text for embedding, and a parallel list of the full chunk info
    chunk_texts_for_embedding = [chunk_info["text"] for chunk_info in all_chunks_with_metadata]
    
    processed_chunks_data_final = []
    for i, chunk_info in enumerate(all_chunks_with_metadata):
        processed_chunks_data_final.append({
            "id": f"france_land_chunk_{i}", # Global unique ID
            "text": chunk_info["text"],
            "metadata": chunk_info["metadata"]
        })

    embedding_model = get_embedding_model()
    chunk_embeddings = create_embeddings(chunk_texts_for_embedding, embedding_model)

    if chunk_embeddings.size == 0:
        print("No embeddings were generated. Cannot build FAISS index.")
        return

    build_and_save_faiss_index(chunk_embeddings, processed_chunks_data_final)
    
    print("Ingestion pipeline completed successfully for all URLs.")
    print(f"FAISS index saved to: {FAISS_INDEX_PATH}")
    print(f"Chunk data saved to: {CHUNKS_DATA_PATH}")

if __name__ == "__main__":
    run_ingestion_pipeline()