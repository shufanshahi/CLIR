import requests
from lxml import etree
from lxml import html
import json
from urllib.parse import urljoin

# URLs to scrape
sitemap_urls = [
    'https://dailyasianage.com/sitemap.xml'

]

all_links = []
news_links = set()

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

# Now, for each link in all_links, fetch and extract news hrefs
for i, page_url in enumerate(all_links):
    if len(news_links) >= 600:
        break
    try:
        print(f"Fetching page {i+1}/{len(all_links)}: {page_url}")
        response = requests.get(page_url, timeout=10)
        response.raise_for_status()
        
        # Parse the HTML content
        tree = html.fromstring(response.content)
        
        # Extract all href attributes from a tags
        href_elements = tree.xpath('//a/@href')
        
        for href in href_elements:
            if '/news/' in href.lower():
                absolute_url = urljoin(page_url, href)
                news_links.add(absolute_url)
                if len(news_links) >= 600:
                    break
        
        print(f"Found {len(href_elements)} links on page, added news links so far: {len(news_links)}")
        
    except Exception as e:
        print(f"Error processing {page_url}: {str(e)}")

# Convert set to list and limit to 600
news_links_list = list(news_links)[:600]

# Save to JSON file
output_file = 'links.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(news_links_list, f, indent=2, ensure_ascii=False)

print(f"\nTotal news links found: {len(news_links_list)}")
print(f"Links saved to {output_file}")
