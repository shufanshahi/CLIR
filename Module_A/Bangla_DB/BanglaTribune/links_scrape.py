import requests
from lxml import etree
import json

# URLs to scrape
sitemap_urls = [
    'https://www.banglatribune.com/archive_2025-04-01.xml',
    'https://www.banglatribune.com/archive_2025-04-02.xml',
    'https://www.banglatribune.com/archive_2025-04-03.xml',
    'https://www.banglatribune.com/archive_2025-04-04.xml'
]

all_links = []

# Scrape each sitemap
for url in sitemap_urls:
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
            if link:
                all_links.append(link)
                print(f"Found link: {link}")
        
        print(f"Found {len(loc_elements)} links from {url}")
        
    except Exception as e:
        print(f"Error processing {url}: {str(e)}")

# Save to JSON file
output_file = 'links.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(all_links, f, indent=2, ensure_ascii=False)

print(f"\nTotal links found: {len(all_links)}")
print(f"Links saved to {output_file}")
