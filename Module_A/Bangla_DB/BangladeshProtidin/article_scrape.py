import requests
from lxml import html
import json
import time
import re
from datetime import datetime, timezone, timedelta

# Load links from links.json
with open('links.json', 'r', encoding='utf-8') as f:
    links = json.load(f)

# Select first 300 and last 300 links, removing duplicates
first_300 = links[:300]
last_300 = links[-300:]
combined_links = first_300 + last_300

# Remove duplicates while preserving order, but keep track of original position
seen = set()
links = []
link_origins = []  # Track if link is from first_300 or last_300
for link in combined_links:
    if link not in seen:
        seen.add(link)
        links.append(link)
        # Determine if this link was from first_300 or last_300
        if link in first_300:
            link_origins.append('first')
        else:
            link_origins.append('last')

articles = []

# Scrape each link
for i, url in enumerate(links, 1):
    try:
        print(f"Scraping {i}/{len(links)}: {url}")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML content
        tree = html.fromstring(response.content)
        
        # Extract h1 (title)
        h1_elements = tree.xpath('//h1')
        title = h1_elements[0].text_content().strip() if h1_elements else ''
        
        # Set date based on whether link is from first 300 or last 300
        origin = link_origins[i-1]
        if origin == 'first':
            # First 300: 2025-12-30 at midnight with +06:00 timezone
            date = '2025-12-30T00:00:00+06:00'
        else:
            # Last 300: 2025-12-02 at midnight with +06:00 timezone
            date = '2025-12-02T00:00:00+06:00'
        
        # Extract p tags from specific article xpath
        p_elements = tree.xpath('/html/body/main/section/div[2]/div[1]/div[3]/article//p')
        body_paragraphs = [p.text_content().strip() for p in p_elements if p.text_content().strip()]
        body = ' '.join(body_paragraphs)
        
        # Remove copyright text from body
        body = re.sub(r'Copyright © \d{4} Bangladesh Pratidin', '', body)
        body = body.strip()
        
        # Create article object
        article = {
            'title': title,
            'body': body,
            'url': url,
            'date': date,
            'language': 'en'
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
