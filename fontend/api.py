import requests

from . import models


class Request_URL:
    url: str
    method: str

    def __init__(self, url: str, method: str):
        self.url = url
        self.method = method


class API:
    request_url: Request_URL
    params: dict = {}
    body: dict = {}

    def __init__(self, request_url: Request_URL, params: dict = {}, body: dict = {}):
        self.request_url = request_url
        self.params = params
        self.body = body

    def make_request(self):
        return requests.request(
            method=self.request_url.method,
            url=self.request_url.url,
            params=self.params,
            data=self.body,
        )


class API_LLM:
    host: str = "http://localhost:8000"
    
    FEATURES = {
        'send_message':'send_message'
    }
    
    def __init__(self, host):
        self.host = host

    def make_request(self, feature_name: str, body: any):
        if feature_name == self.FEATURES['send_message']:
            return self.send_message(body)
        raise Exception(f"Not found feature {feature_name}")

    def send_message(self, message: models.Message):
        request_url = Request_URL(url=f"{self.host}/chat/send")
        api = API(request_url, body=message.to_dict())
        return api.make_request()
