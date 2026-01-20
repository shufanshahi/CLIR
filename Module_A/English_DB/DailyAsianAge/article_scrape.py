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
        
        # Extract h1 (title)
        h1_elements = tree.xpath('//h1')
        title = h1_elements[0].text_content().strip() if h1_elements else ''
        
        # Extract date and time from specific xpath text node
        date_texts = tree.xpath('/html/body/div[3]/div[2]/div[2]/div/p/text()')
        date_str = ''.join(date_texts).strip() if date_texts else ''
        
        # Clean up the date string - remove non-breaking spaces and other whitespace
        date_str = re.sub(r'[\xa0\s]+', ' ', date_str).strip()
        
        # Parse date string to datetime
        try:
            # Try multiple date formats
            dt = None
            for fmt in ['%I:%M %p, %d %B %Y', '%d %B %Y %I:%M %p']:
                try:
                    dt = datetime.strptime(date_str, fmt)
                    break
                except ValueError:
                    continue
            
            if dt:
                # Assume timezone +06:00
                dt = dt.replace(tzinfo=timezone(timedelta(hours=6)))
                date = dt.isoformat()
            else:
                date = ''
        except Exception:
            date = ''
        
        # Extract body from div id="news-detail-content"
        body_element = tree.xpath('//div[@id="news-detail-content"]')
        body = body_element[0].text_content().strip() if body_element else ''
        
        # Clean up the body text
        # Remove the specific jQuery AJAX code
        body = re.sub(r'\$\.ajax\(\{type:\'post\',url:\\?"https://dailyasianage\.com/ajax/ajax\.php\\?",data:\'latestNews=\'\+1,dataType:\'text\',success:function\(data\)\{\$\(\\?"#latestNews\\?"\)\.html\(data\);\}\}\);', '', body)
        
        # Remove other JavaScript patterns
        body = re.sub(r'\$\.ajax\({[^}]*}\);', '', body)
        body = re.sub(r'<script[^>]*>.*?</script>', '', body, flags=re.DOTALL)
        
        # Remove common unwanted patterns
        body = re.sub(r'Latest News\s*', '', body)
        body = re.sub(r'>> Source:.*$', '', body)
        
        # Remove excessive whitespace and newlines
        body = re.sub(r'\n\s*\n+', '\n', body)
        body = re.sub(r'\s+', ' ', body)
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
