import scrapy
import json
from scrapy.http import HtmlResponse


class DecryptSpider(scrapy.Spider):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.page = 0

    name = 'decrypt'
    allowed_domains = ['decrypt.co']
    start_urls = ['https://api.decrypt.co/content-elasticsearch/posts?_minimal=true&category=news&lang=en-US&offset=0'
                  '&order=desc&orderby=date&per_page=1000&type=post']

    def parse(self, response):
        data = json.loads(response.body)
        for item in data:
            result = HtmlResponse(url="", body=item["content"]["rendered"], encoding='utf-8')
            content = ''.join(result.css("span::text").getall()).replace('\xa0', ' ')
            output = {
                "date": item["date_gmt"],
                "title": item["title"]["rendered"],
                "link": item["link"],
                "content": content,
            }
            yield output
        self.page += 1000
        next_page = f'https://api.decrypt.co/content-elasticsearch/posts?_minimal=true&category=news&lang=en-US&offset={self.page}&order=desc&orderby=date&per_page=1000&type=post'
        if self.page <= 5000:
            yield response.follow(next_page, callback=self.parse)
