from os import urandom
from bson import ObjectId
from flask_login import UserMixin
from itsdangerous import TimedJSONWebSignatureSerializer as Serializer

from application import app, db
from application.classes.session import Session

class User(UserMixin):
    def __init__(self, id:str=None, email:str=None, password:str=None, _is_active:bool=True, sessions:str=[]):
        self.id = str(id)
        self.email = email
        self.password = password
        self._is_active = _is_active
        self.sessions = sessions

    def __repr__(self):
        return f"User('{str(self.id)}')"
    
    @property
    def is_active(self):
        return self._is_active

    @is_active.setter
    def is_active(self, value):
        self._is_active = value
    
    def get_id(self):
        print(self.id)
        return self.id
    
    def to_dict(self):
        dictionary = {
            "id": str(self.id),
            "email": self.email,
            "password": self.password,
            "_is_active": self._is_active,
            "sessions": [session.to_dict() for session in self.sessions]
        }
        return dictionary

    @staticmethod
    def from_dict(dictionary:dict):
        if dictionary == None:
            return None
        user = User(str(dictionary.get("id")),
                    dictionary.get("email"),
                    dictionary.get("password"),
                    dictionary.get("_is_active"),
                    [Session.from_dict(session) for session in dictionary.get('sessions')]
            )
        return user
    
    def add(self):
        db.users.insert(self.to_dict())
    
    @staticmethod
    def get_by_id(id: str):
        return User.from_dict(db.users.find_one({"id": str(id)}))
    
    @staticmethod
    def get_by_email(email: str):
        return User.from_dict(db.users.find_one({"email": str(email)}))

    def get_session_by_id(self, session_id: str):
        sessions = [session for session in self.sessions if session.id == session_id]
        return sessions[0] if len(sessions) > 0 else None

    def add_session(self, session: Session):
        db.users.update({"id": self.id}, {"$push": {"sessions": session.to_dict()}})

    def delete_session(self, session_id: str):
        db.users.update({"id": self.id}, {"$pull": {"sessions": {"id": session_id}}})
    
    def update_session(self, session_id, **kwargs):
        for key, value in kwargs.items():
            db.users.update({"id": self.id, "sessions.id": session_id}, {"$set": {f"sessions.$.{key}": value}}, False, True)

    def update_email(self, email:str):
        db.users.update({"id": self.id}, {'$set' : {"email":email}})

    def update_password(self, password: str):
        db.users.update({"id": self.id}, {'$set' : {"password":password}})

    @staticmethod
    def get_total_users():
        return db.users.count()