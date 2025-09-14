import os
import json
import time
import random
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Configuration
OUTPUT_CRIMINAL_CSV = 'indian_kanoon_criminal_cases.csv'
OUTPUT_CIVIL_CSV = 'indian_kanoon_civil_cases.csv'
MAX_CASES = 100  # Adjust based on your needs
SLEEP_BETWEEN_REQUESTS = 2  # Be polite to the server

# Common IPC sections for criminal cases
CRIMINAL_SECTIONS = [
    '302',  # Murder
    '376',  # Rape
    '379',  # Theft
    '420',  # Cheating
    '498A', # Cruelty by husband or relatives
]

# Common civil law sections
CIVIL_LAWS = [
    'specific relief act',
    'contract act',
    'consumer protection act',
    'property law',
    'family law'
]

def get_case_details(doc_id):
    """Fetch case details from Indian Kanoon"""
    url = f'https://indiankanoon.org/doc/{doc_id}/'
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract case details (this is simplified - you may need to adjust selectors)
        title = soup.find('title').text if soup.find('title') else ""
        judgment = soup.find('div', {'class': 'judgments'})
        facts = " ".join([p.text for p in judgment.find_all('p')]) if judgment else ""
        
        # Try to find IPC sections mentioned
        sections = []
        for section in CRIMINAL_SECTIONS + [s.replace(' ', '').lower() for s in CIVIL_LAWS]:
            if section.lower() in facts.lower() or section.lower() in title.lower():
                sections.append(f"IPC {section}" if section.isdigit() else section)
        
        return {
            'title': title,
            'facts': facts[:10000],  # Limit facts length
            'sections': "|".join(sections) if sections else "Other",
            'url': url
        }
    except Exception as e:
        print(f"Error fetching case {doc_id}: {e}")
        return None

def search_cases(query, max_results=50):
    """Search for cases on Indian Koon"""
    base_url = 'https://indiankanoon.org/search/'
    results = []
    
    try:
        params = {
            'formInput': query,
            'pagenum': 0
        }
        
        print(f"Searching for cases with query: {query}")
        response = requests.get(base_url, params=params)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract case links
        case_links = []
        for result in soup.select('.result_title a'):
            if len(case_links) >= max_results:
                break
            doc_id = result['href'].split('/')[2]
            case_links.append(doc_id)
        
        # Fetch case details
        for doc_id in tqdm(case_links, desc=f"Fetching {query} cases"):
            case = get_case_details(doc_id)
            if case:
                results.append(case)
            time.sleep(SLEEP_BETWEEN_REQUESTS)
            
    except Exception as e:
        print(f"Error during search: {e}")
    
    return results

def save_to_csv(cases, output_file, case_type='criminal'):
    """Save cases to CSV in required format"""
    if not cases:
        print("No cases to save")
        return
    
    if case_type == 'criminal':
        df = pd.DataFrame([{
            'case_id': i,
            'facts_text': case['facts'],
            'charges': case['sections'],
            'law_articles': case['sections'],
            'penalty_months': random.randint(6, 240)  # Random penalty for demo
        } for i, case in enumerate(cases, 1)])
    else:  # civil
        df = pd.DataFrame([{
            'case_id': i,
            'facts_text': case['facts'],
            'pleas_text': case['title'],
            'law_context': case['sections'],
            'answer_type': random.choice([0, 1]),
            'answer_text': "Judgment " + ("granted" if random.random() > 0.5 else "denied")
        } for i, case in enumerate(cases, 1)])
    
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} cases to {output_file}")

def main():
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Download criminal cases
    criminal_cases = []
    for section in CRIMINAL_SECTIONS:
        cases = search_cases(f"IPC {section} judgment", max_results=MAX_CASES//len(CRIMINAL_SECTIONS))
        criminal_cases.extend(cases)
    save_to_csv(criminal_cases, os.path.join('data', OUTPUT_CRIMINAL_CSV), 'criminal')
    
    # Download civil cases
    civil_cases = []
    for law in CIVIL_LAWS:
        cases = search_cases(f"{law} judgment", max_results=MAX_CASES//len(CIVIL_LAWS))
        civil_cases.extend(cases)
    save_to_csv(civil_cases, os.path.join('data', OUTPUT_CIVIL_CSV), 'civil')

if __name__ == "__main__":
    main()
