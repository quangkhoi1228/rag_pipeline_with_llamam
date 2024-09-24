import copy
import requests

import models


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
    headers: dict = {"Content-Type": "application/json"}

    def __init__(self, request_url: Request_URL, params: dict = {}, body: dict = {}):
        self.request_url = request_url
        self.params = params
        self.body = body

    def set_headers(self, headers: dict):
        self.headers = headers

    async def make_request(self):
        return requests.request(
            headers=self.headers,
            method=self.request_url.method,
            url=self.request_url.url,
            params=self.params,
            json=self.body,
        ).json()


class API_LLM:
    host: str = "http://localhost:8000"

    FEATURES = {
        "send_message": "send_message",
        "regenerate_response": "regenerate_response",
        "clear_chat": "clear_chat"
    }

    def __init__(self, host: str = "http://localhost:8000"):
        self.host = host

    async def make_request(self, feature_name: str, body: any):

        if feature_name == self.FEATURES["send_message"]:
            return await self.send_message(body)

        elif feature_name == self.FEATURES["regenerate_response"]:
            return await self.regenerate_response(body)
        
        elif feature_name == self.FEATURES["clear_chat"]:
            return await self.clear_chat(body)

        raise Exception(f"Not found feature {feature_name}")

    async def send_message(self, message: models.Message):
        request_url = Request_URL(url=f"{self.host}/chat/send", method="POST")
        api = API(request_url, body=copy.deepcopy(message).to_dict())
        return await api.make_request()

    async def regenerate_response(self, message: models.Message):
        request_url = Request_URL(url=f"{self.host}/chat/regenerate", method="POST")
        api = API(request_url, body=copy.deepcopy(message).to_dict())
        return await api.make_request()
    
    async def clear_chat(self):
        request_url = Request_URL(url=f"{self.host}/chat/clear", method="DELETE")
        api = API(request_url)
        return await api.make_request()
