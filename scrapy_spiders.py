import scrapy
from datetime import datetime
from dateutil import parser
import csv
import os

class MalayMailSpider(scrapy.Spider):
    name = "malaymail"
    allowed_domains = ["malaymail.com"]
    start_urls = ["https://malaymail.com/morearticles/?pgno=1"]
    page_count = 0
    max_page_limit = 1  # Set the maximum number of pages to scrape

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

            yield {
                'Date': formatted_date_time,
                'Title': response.css('h1.article-title::text').get(),
                'Description': response.css('div.article-body p::text').getall()
            }

            # Save the scraped data to CSV
            self.save_to_csv('malaymail.csv', formatted_date_time, response.css('h1.article-title::text').get(), response.css('div.article-body p::text').getall())

    def save_to_csv(self, file_name, date, title, description):
        file_path = os.path.join('C:/Users/Windows/port', file_name)
        header = ['Date', 'Title', 'Description']

        # Check if the file already exists
        file_exists = os.path.isfile(file_path)

        with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write header if the file is empty
            if not file_exists:
                csv_writer.writerow(header)

            csv_writer.writerow([date, title, description])

class TheStarSpider(scrapy.Spider):
    name = "thestar"
    allowed_domains = ["thestar.com.my"]
    start_urls = ["https://www.thestar.com.my/news/latest/"]
    page_count = 0
    max_page_limit = 1  # Set the maximum number of pages to scrape

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
            yield {
                'Date': formatted_date,
                'Title': response.css('div.headline h1::text').get(default='').strip(),
                'Description': response.css('div.story p::text').getall()
            }

            # Save the scraped data to CSV
            self.save_to_csv('thestar.csv', formatted_date, response.css('div.headline h1::text').get(default='').strip(), response.css('div.story p::text').getall())

    def save_to_csv(self, file_name, date, title, description):
        file_path = os.path.join('C:/Users/Windows/port', file_name)
        header = ['Date', 'Title', 'Description']

        # Check if the file already exists
        file_exists = os.path.isfile(file_path)

        with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write header if the file is empty
            if not file_exists:
                csv_writer.writerow(header)

            csv_writer.writerow([date, title, description])


class BernamaSpider(scrapy.Spider):
    name = "bernama"
    allowed_domains = ["bernama.com"]
    start_urls = ["https://bernama.com/en/"]
    page_count = 0
    max_page_limit = 1  # Set the maximum number of pages to scrape

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
                yield {
                    'Date': formatted_date,
                    'Title': response.css('div h1.h2::text').get().strip(),
                    'Description': description
                }

                # Save the scraped data to CSV
                self.save_to_csv('bernama.csv', formatted_date, response.css('div h1.h2::text').get().strip(), description)
        else:
            self.logger.warning("Invalid or missing date string.")

    def save_to_csv(self, file_name, date, title, description):
        file_path = os.path.join('C:/Users/Windows/port', file_name)
        header = ['Date', 'Title', 'Description']

        # Check if the file already exists
        file_exists = os.path.isfile(file_path)

        with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)

            # Write header if the file is empty
            if not file_exists:
                csv_writer.writerow(header)

            csv_writer.writerow([date, title, description])
