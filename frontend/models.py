import datetime

class Message():
    message: str = 'Quyết định 720/QĐ-CTN năm 2020'  # Auto-generated UUID
    history_count: int = 6
    faq_id: str = ''
    
    def __init__(self, message:str, history_count: int, faq_id: str = ''):
        self.message = message
        self.history_count = history_count
        self.faq_id = faq_id
        
    def set_faq_id(self, faq_id: str):
        self.faq_id = faq_id
        
    def to_dict(self):
        return {
            "message": self.message,
            "history_count": self.history_count,
            "faq_id": self.faq_id
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
    faq_id: str = ''
    faq_pool_id: str = ''
    
    def __init__(self, response:Assistant_Respone, references: list[Assistant_Ref], faq_id: str = '', faq_pool_id: str = ''):
        self.response = response
        self.references = references
        self.faq_id = faq_id
        self.faq_pool_id = faq_pool_id
        
class Feedback():
    faq_id: str =  "",
    faq_pool_id: str = "",
    feedback: str = "good"
    
    def __init__(self, faq_id: str, faq_pool_id: str, feedback: str):
        self.faq_id = faq_id
        self.faq_pool_id = faq_pool_id
        self.feedback = feedback
    
    def to_dict(self):
        return {
            "faq_id": self.faq_id,
            "faq_pool_id": self.faq_pool_id,
            "feedback": self.feedback
        }