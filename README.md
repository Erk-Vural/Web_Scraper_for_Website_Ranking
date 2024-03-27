# Web Scraper for Website Ranking
 scraper utilizes Flask for the backend and implements a ranking system based on website similarity using keywords. The Flask web application provides a user-friendly interface for interacting with the scraper and viewing ranked websites.

## Features

- **Web Scraping**: Scrapes websites to extract relevant keywords for ranking.
- **Keyword-Based Ranking**: Ranks websites based on similarity using extracted keywords.
- **Flask Web Application**: Provides a web interface for users to interact with the scraper and view ranked websites.
- **Customizable**: Easily configurable for different scraping and ranking requirements.

## Requirements

- Python 3.x
- Flask
- BeautifulSoup (for web scraping)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/web_scraper.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask web application:
   ```bash
   python app.py
   ```

4. Access the application at `http://localhost:5000` in your web browser.

## Usage

1. Initiate the scraping process.
2. Wait for the scraper to extract keywords and rank the website based on similarity.
3. View the ranked websites along with their similarity scores.
4. Repeat the process for additional websites as needed.
