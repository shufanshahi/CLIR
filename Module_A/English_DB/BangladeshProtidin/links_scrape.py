import requests
from lxml import etree
import json

# URLs to scrape
sitemap_urls = [
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-30/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-29/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-28/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-27/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-26/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-25/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-24/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-23/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-22/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-21/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-20/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-19/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-18/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-17/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-16/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-15/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-14/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-13/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-12/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-11/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-10/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-09/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-08/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-07/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-06/sitemap.xml',
    'https://en.bd-pratidin.com/daily-sitemap/2025-12-05/sitemap.xml'
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
