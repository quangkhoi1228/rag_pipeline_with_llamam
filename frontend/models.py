import datetime

class Message():
    message: str = 'Quyết định 720/QĐ-CTN năm 2020'  # Auto-generated UUID
    history_count: int = 6
    
    def __init__(self, message:str, history_count: int):
        self.message = message
        self.history_count = history_count
        
    def to_dict(self):
        return {
            "message": self.message,
            "history_count": self.history_count
        }

class Assistant_Respone():
    message:str
    sender: str
    created_date: datetime
    
    def __init__(self, message:str, sender: str, created_date:datetime):
        self.message = message
        self.sender = sender
        self.created_date = created_date

class Assistant_Ref():
    url:str
    title: str
    
    def __init__(self, url:str, title: str):
        self.url = url
        self.title = title

class Assistant_Message():
    response: Assistant_Respone
    references: list[Assistant_Ref]
    
    def __init__(self, response:Assistant_Respone, references: list[Assistant_Ref]):
        self.response = response
        self.references = references