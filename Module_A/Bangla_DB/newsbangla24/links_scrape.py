import requests
from lxml import etree
import json
from datetime import datetime, timedelta

# Generate sitemap URLs for the last 60 days
today = datetime.now()
sitemap_urls = []
for i in range(60):
    date = today - timedelta(days=i)
    date_str = date.strftime('%Y-%m-%d')
    url = f'https://www.newsbangla24.com/daily-sitemap/{date_str}/sitemap.xml'
    sitemap_urls.append((url, date_str))

all_links = []

# Scrape each sitemap
for url, date_str in sitemap_urls:
    try:
        print(f"Fetching: {url}")
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the XML content
        root = etree.fromstring(response.content)
        
        # Extract all <loc> elements (the actual URLs)
        # Handle XML namespace if present
        namespaces = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
        
        # Try with namespace first
        loc_elements = root.xpath('//ns:loc', namespaces=namespaces)
        
        # If no results, try without namespace
        if not loc_elements:
            loc_elements = root.xpath('//loc')
        
        for loc in loc_elements:
            link = loc.text.strip() if loc.text else ''
            if link and '/news/' in link:
                all_links.append({'url': link, 'date': date_str})
                print(f"Found news link: {link}")
        
        print(f"Found {len(loc_elements)} links from {url}, filtered to {len([l for l in loc_elements if l.text and '/news/' in l.text])} news links")
        
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")

# Save to JSON file
output_file = 'links.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_links, f, indent=2, ensure_ascii=False)

print(f"\nTotal news links found: {len(all_links)}")
print(f"Links saved to {output_file}")
