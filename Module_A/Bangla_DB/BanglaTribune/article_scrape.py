import requests
from lxml import html
import json
import time
import re

# Load links from links.json
with open('links.json', 'r', encoding='utf-8') as f:
    links = json.load(f)

# Select first 300 and last 300 links, removing duplicates
first_300 = links[:300]
last_300 = links[-300:]
combined_links = first_300 + last_300

# Remove duplicates while preserving order
seen = set()
links = []
for link in combined_links:
    if link not in seen:
        seen.add(link)
        links.append(link)

articles = []

# Scrape each link
for i, url in enumerate(links, 1):
    try:
        print(f"Scraping {i}/{len(links)}: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML content
        tree = html.fromstring(response.content)
        
        # Extract title from first row
        title_elements = tree.xpath('(//div[contains(@class, "row")]//h1)[1]')
        title = title_elements[0].text_content().strip() if title_elements else ''
        
        # Extract datetime from tts_time content attribute
        date_elements = tree.xpath('//span[@class="tts_time"]/@content')
        date = date_elements[0] if date_elements else ''
        
        # Extract all p tags from first row detail_holder class
        body_elements = tree.xpath('(//div[contains(@class, "detail_holder")])[1]//p')
        body_paragraphs = [p.text_content().strip() for p in body_elements if p.text_content().strip()]
        body = ' '.join(body_paragraphs)
        
        # Clean unwanted patterns
        body = re.sub(r'jwARI\.fetch\( \$\( ".*?" \) \);', '', body)
        
        # Create article object
        article = {
            'title': title,
            'body': body,
            'url': url,
            'date': date,
            'language': 'bn'
        }
        
        articles.append(article)
        print(f"✓ Successfully scraped: {title[:50]}...")
        
        # Be polite to the server
        time.sleep(1)
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Network error scraping {url}: {str(e)}")
        continue
    except Exception as e:
        print(f"✗ Error scraping {url}: {str(e)}")
        continue

# Save to JSON file
output_file = 'articles.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(articles, f, indent=2, ensure_ascii=False)

print(f"\nTotal articles scraped: {len(articles)}")
print(f"Articles saved to {output_file}")
