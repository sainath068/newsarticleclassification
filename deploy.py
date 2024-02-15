import scrapy
from datetime import datetime
from dateutil import parser
import os
import pandas as pd
import streamlit as st
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

class MalayMailSpider(scrapy.Spider):
    name = "malaymail"
    allowed_domains = ["malaymail.com"]
    start_urls = ["https://malaymail.com/morearticles/?pgno=1"]
    page_count = 0
    max_page_limit = 1  # Set the maximum number of pages to scrape
    output_data = []

    def parse(self, response):
        news = response.css('div.article-item')

        for article in news:
            relative_url = article.css('h2 a::attr(href)').get()
            absolute_url = response.urljoin(relative_url)
            yield response.follow(absolute_url, callback=self.parse_article)

        self.page_count += 1

        # Modify the condition to allow following pages up to the maximum limit
        if self.page_count < self.max_page_limit:
            next_page_number = self.page_count + 1
            next_page_url = f'https://malaymail.com/morearticles/?pgno={next_page_number}'
            yield response.follow(next_page_url, callback=self.parse)

    def parse_article(self, response):
        date_time_str = response.css('div.article-date::text').get()
        date_time_obj = parser.parse(date_time_str)

        if date_time_obj.date() == datetime.now().date():
            formatted_date_time = date_time_obj.strftime('%A, %d %b %Y %I:%M %p %Z')

            data = {
                'Date': formatted_date_time,
                'Title': response.css('h1.article-title::text').get(),
                'Description': response.css('div.article-body p::text').getall()
            }

            self.output_data.append(data)

class TheStarSpider(scrapy.Spider):
    name = "thestar"
    allowed_domains = ["thestar.com.my"]
    start_urls = ["https://www.thestar.com.my/news/latest/"]
    page_count = 0
    max_page_limit = 1  # Set the maximum number of pages to scrape
    output_data = []

    def parse(self, response):
        articles = response.css('div.timeline-content')

        for article in articles:
            relative_url = article.css('h2.f18 a::attr(href)').get()
            absolute_url = response.urljoin(relative_url)
            yield response.follow(absolute_url, callback=self.parse_article)

        self.page_count += 1

        # Modify the condition to allow following pages up to the maximum limit
        if self.page_count < self.max_page_limit:
            next_page_number = self.page_count + 1
            next_page_url = f'https://www.thestar.com.my/news/latest?pgno={next_page_number}#Latest'
            yield response.follow(next_page_url, callback=self.parse)

    def parse_article(self, response):
        date_str = response.css('li p.date ::text').get().strip()
        date_obj = datetime.strptime(date_str, '%A, %d %b %Y')
        formatted_date = date_obj.strftime('%d %B %Y')

        if date_obj.date() == datetime.now().date():
            data = {
                'Date': formatted_date,
                'Title': response.css('div.headline h1::text').get(default='').strip(),
                'Description': response.css('div.story p::text').getall()
            }

            self.output_data.append(data)

class BernamaSpider(scrapy.Spider):
    name = "bernama"
    allowed_domains = ["bernama.com"]
    start_urls = ["https://bernama.com/en/"]
    page_count = 0
    max_page_limit = 1  # Set the maximum number of pages to scrape
    output_data = []

    def parse(self, response):
        articles = response.css('div.mb-lg-0')

        for article in articles:
            relative_url = article.css('h6 a::attr(href)').get()
            absolute_url = response.urljoin(relative_url)
            yield response.follow(absolute_url, callback=self.parse_article)

        self.page_count += 1

        # Modify the condition to allow following pages up to the maximum limit
        if self.page_count < self.max_page_limit:
            next_page_number = self.page_count + 1
            next_page_url = f'https://www.bernama.com/en/general/index.php?page={next_page_number}'
            yield response.follow(next_page_url, callback=self.parse)

    def parse_article(self, response):
        # Get the raw date string
        date_str = response.xpath('//*[@id="body-row"]/div[2]/div[2]/div/div[1]/div[3]/div[2]/div/text()').get()

        # Strip extra spaces and characters
        date_str = date_str.strip()

        # Perform pre-validation on the date string
        if date_str:
            # Attempt to parse the date string into a datetime object
            date_obj = datetime.strptime(date_str, '%d/%m/%Y %I:%M %p')

            # Format the date into the desired format
            formatted_date = date_obj.strftime('%d %B %Y %I:%M %p')

            # Extracting description as a single string
            description = ' '.join(response.css('div.text-dark p::text').getall())

            # Check if the parsed date is today's date
            if date_obj.date() == datetime.now().date():
                data = {
                    'Date': formatted_date,
                    'Title': response.css('div h1.h2::text').get().strip(),
                    'Description': description
                }

                self.output_data.append(data)
        else:
            self.logger.warning("Invalid or missing date string.")

# TextAnalyzer class for sentiment analysis and summarization
class TextAnalyzer:
    SENTIMENT_THRESHOLD = 0.5
    POPULARITY_THRESHOLD = 1500

    def __init__(self):
        # Initializing sentiment and summarization models
        self.tokenizer_sentiment, self.model_sentiment = self.initialize_sentiment_model()
        self.tokenizer_summarization, self.model_summarization = self.initialize_summarization_model()

    # ... (same as your provided TextAnalyzer class)

# Streamlit app
def main():
    st.title("News Article Classification and Summarization")

    malaymail_selected = st.checkbox("Malay Mail")
    thestar_selected = st.checkbox("The Star")
    bernama_selected = st.checkbox("Bernama")

    if st.button("Scrape News"):
        st.text("Scraping... Please wait!")

        scraped_data = []

        if malaymail_selected:
            scraped_data.extend(scrape_website(MalayMailSpider, "Malay Mail"))

        if thestar_selected:
            scraped_data.extend(scrape_website(TheStarSpider, "The Star"))

        if bernama_selected:
            scraped_data.extend(scrape_website(BernamaSpider, "Bernama"))

        st.text("Scraping finished!")

        display_scraped_data(scraped_data)

    summarization_checkbox = st.checkbox("Text Summarization")

    if summarization_checkbox:
        if st.button("Summarize Text"):
            st.text("Performing Text Summarization... Please wait!")

            text_analyzer = TextAnalyzer()

            if malaymail_selected:
                summarize_and_display_data(text_analyzer, MalayMailSpider.output_data, "Malay Mail", "Description")

            if thestar_selected:
                summarize_and_display_data(text_analyzer, TheStarSpider.output_data, "The Star", "Description")

            if bernama_selected:
                summarize_and_display_data(text_analyzer, BernamaSpider.output_data, "Bernama", "Description")

            st.text("Text Summarization finished!")

    sentiment_classification_checkbox = st.checkbox("Sentiment Classification")

    if sentiment_classification_checkbox:
        if st.button("Classify Sentiment"):
            st.text("Performing Sentiment Classification... Please wait!")

            text_analyzer = TextAnalyzer()

            if malaymail_selected:
                classify_and_display_data(text_analyzer, MalayMailSpider.output_data, "Malay Mail", "Description")

            if thestar_selected:
                classify_and_display_data(text_analyzer, TheStarSpider.output_data, "The Star", "Description")

            if bernama_selected:
                classify_and_display_data(text_analyzer, BernamaSpider.output_data, "Bernama", "Description")

            st.text("Sentiment Classification finished!")

def scrape_website(spider_class, website_name):
    process = CrawlerProcess(get_project_settings())
    process.crawl(spider_class)
    process.start()

    scraped_data = spider_class.output_data

    if scraped_data:
        st.success(f"Successfully scraped data from {website_name}")
    else:
        st.warning(f"No data scraped from {website_name}")

    return scraped_data

def display_scraped_data(scraped_data):
    st.subheader("Scraped Data:")
    st.write(pd.DataFrame(scraped_data))

def summarize_and_display_data(text_analyzer, data, website, text_column):
    summarized_data = []

    for item in data:
        title = item['Title']
        description = item[text_column]

        if description:
            summary = text_analyzer.summarize_text(description)
            sentiment = text_analyzer.analyze_sentiment(summary)

            summarized_data.append({
                'Title': title,
                'Original Description': description,
                'Summarized Description': summary,
                'Sentiment': sentiment
            })

    st.subheader(f"{website} Summarized Data:")
    st.write(pd.DataFrame(summarized_data))

def classify_and_display_data(text_analyzer, data, website, text_column):
    classified_data = []

    for item in data:
        title = item['Title']
        description = item[text_column]

        if description:
            sentiment = text_analyzer.analyze_sentiment(description)

            classified_data.append({
                'Title': title,
                'Description': description,
                'Sentiment': sentiment
            })

    st.subheader(f"{website} Classified Data:")
    st.write(pd.DataFrame(classified_data))

if __name__ == "__main__":
    main()
