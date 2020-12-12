from os import urandom 
from bson import ObjectId

from application import db

class Session:
    def __init__(self, id:str, start_time=None, end_time=None, total_time=None, focus:list=None, heatmap:list=None, mood:list=None, overall_mood:str=None, strain:list=None, overall_strain:int=None):
        self.id = str(id)
        self.start_time = start_time
        self.end_time = end_time
        self.total_time = total_time
        self.focus = focus 
        self.heatmap = heatmap
        self.mood = mood 
        self.overall_mood = overall_mood
        self.strain = strain
        self.overall_strain = overall_strain

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(dictionary:dict):
        return Session( str(dictionary.get("id")),
                        dictionary.get("start_time"),
                        dictionary.get("end_time"),
                        dictionary.get("total_time"),
                        dictionary.get("focus"),
                        dictionary.get("heatmap"),
                        dictionary.get("mood"),
                        dictionary.get("overall_mood"),
                        dictionary.get("strain"),
                        dictionary.get("overall_strain")
            )

    def __repr__(self):
        return f"Session('{str(self.id)}')"