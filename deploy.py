import streamlit as st
import pandas as pd
import multiprocessing
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from transformers import T5Tokenizer, T5ForConditionalGeneration, DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import re

# Import your Scrapy spiders
from scrapy_spiders import MalayMailSpider, TheStarSpider, BernamaSpider

class TextAnalyzer:
    SENTIMENT_THRESHOLD = 0.5
    POPULARITY_THRESHOLD = 1500

    def __init__(self):
        # Initializing sentiment and summarization models
        self.tokenizer_sentiment, self.model_sentiment = self.initialize_sentiment_model()
        self.tokenizer_summarization, self.model_summarization = self.initialize_summarization_model()

    def initialize_sentiment_model(self):
        # Initializing sentiment analysis model
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        return tokenizer, model

    def initialize_summarization_model(self):
        # Initializing text summarization model
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")
        return tokenizer, model

    def get_sentiment_score(self, text):
        # Get sentiment score for the input text
        inputs_sentiment = self.tokenizer_sentiment.encode(text, return_tensors="pt", max_length=512, truncation=True)
        outputs_sentiment = self.model_sentiment(inputs_sentiment)[0]
        positive_class_index = 1
        sentiment_score = torch.sigmoid(outputs_sentiment[:, positive_class_index]).item()
        return sentiment_score

    def calculate_popularity_metric(self, text):
        # Calculate popularity metric (length of text)
        return len(text)

    def classify_and_summarize_text(self, text):
        # Classify text as hot news and summarize it
        sentiment_score = self.get_sentiment_score(text)
        print(sentiment_score)
        popularity_metric = self.calculate_popularity_metric(text)
        is_hot_news = sentiment_score > self.SENTIMENT_THRESHOLD and popularity_metric > self.POPULARITY_THRESHOLD
        inputs_summarization = self.tokenizer_summarization.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.model_summarization.generate(inputs_summarization, max_length=180, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = self.tokenizer_summarization.decode(summary_ids[0], skip_special_tokens=True)
        summary = re.sub(r'[^a-zA-Z0-9\s]', '', summary)
        print(summary)
        return is_hot_news, summary

def run_spider(spider_class, website):
    # Run Scrapy spider for web scraping
    process = CrawlerProcess(get_project_settings())
    process.crawl(spider_class)
    process.start()

    if website == "Malay Mail" or website == "The Star" or website == "Bernama":
        # Display scraped data in the Streamlit app
        display_data(website)

def scrape_website(website):
    if website == "Malay Mail":
        run_spider(MalayMailSpider, website)

    elif website == "The Star":
        run_spider(TheStarSpider, website)

    elif website == "Bernama":
        run_spider(BernamaSpider, website)

def display_data(website):
    # Read and display scraped data in the Streamlit app
    scraped_data = pd.DataFrame()  # Modify this part to load data directly
    st.subheader(f"{website} Scraped Data")
    st.write(scraped_data)

def save_summarized_data(text_analyzer, data, website):
    # Save and display summarized data in the Streamlit app
    summarized_data = pd.DataFrame()
    summarized_data['Title'] = data['Title']
    summarized_data['Date'] = data['Date']
    summarized_data['Summarized_Text'] = data['Description'].apply(lambda x: text_analyzer.classify_and_summarize_text(x)[1])

    st.subheader(f"{website} Summarized Data")
    st.write(summarized_data)

def classify_and_save_data(text_analyzer, data, website):
    # Classify, save, and display data in the Streamlit app
    classified_data = pd.DataFrame()
    classified_data['Title'] = data['Title']
    classified_data['Date'] = data['Date']
    classified_data['Description'] = data['Description']
    
    classified_data['Is_Hot_News'], classified_data['Summarized_Text'] = list(zip(*data['Description'].apply(lambda x: text_analyzer.classify_and_summarize_text(x))))
    classified_data['Is_Hot_News'] = classified_data['Is_Hot_News'].apply(lambda x: 'Hot News' if x else 'Not Hot News')

    st.subheader(f"{website} Classified Data")
    st.write(classified_data[['Title', 'Date', 'Summarized_Text', 'Is_Hot_News']])

def main():
    # Streamlit app main function
    st.title("News Article Classification")

    malaymail_selected = st.checkbox("Malay Mail")
    thestar_selected = st.checkbox("The Star")
    bernama_selected = st.checkbox("Bernama")

    if st.button("Search"):
        # Web scraping button is clicked
        st.text("Scraping... Please wait!")

        if malaymail_selected:
            # Run web scraping for Malay Mail
            scrape_website("Malay Mail")

        if thestar_selected:
            # Run web scraping for The Star
            scrape_website("The Star")

        if bernama_selected:
            # Run web scraping for Bernama
            scrape_website("Bernama")

        st.text("Scraping finished!")

    summarization_checkbox = st.checkbox("Text Summarization")

    if summarization_checkbox:
        if st.button("Summarize Text"):
            # Summarization button is clicked
            st.text("Performing Text Summarization... Please wait!")

            text_analyzer = TextAnalyzer()

            if malaymail_selected:
                malaymail_data = pd.DataFrame()  # Modify this part to load data directly
                save_summarized_data(text_analyzer, malaymail_data, 'Malay Mail')

            if thestar_selected:
                thestar_data = pd.DataFrame()  # Modify this part to load data directly
                save_summarized_data(text_analyzer, thestar_data, 'The Star')

            if bernama_selected:
                bernama_data = pd.DataFrame()  # Modify this part to load data directly
                save_summarized_data(text_analyzer, bernama_data, 'Bernama')

            st.text("Text Summarization finished!")

    classification_checkbox = st.checkbox("Text Classification")

    if classification_checkbox:
        if st.button("Classify Text"):
            # Classification button is clicked
            st.text("Performing Text Classification... Please wait!")

            text_analyzer = TextAnalyzer()

            if malaymail_selected:
                malaymail_data = pd.DataFrame()  # Modify this part to load data directly
                classify_and_save_data(text_analyzer, malaymail_data, 'Malay Mail')

            if thestar_selected:
                thestar_data = pd.DataFrame()  # Modify this part to load data directly
                classify_and_save_data(text_analyzer, thestar_data, 'The Star')

            if bernama_selected:
                bernama_data = pd.DataFrame()  # Modify this part to load data directly
                classify_and_save_data(text_analyzer, bernama_data, 'Bernama')

            st.text("Text Classification finished!")

if __name__ == "__main__":
    main()
