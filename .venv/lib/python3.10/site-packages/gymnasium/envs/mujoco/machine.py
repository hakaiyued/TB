import abc
import numpy as np

class StationBase(abc.ABC):

    def __init__(self):
        is_working = False
        is_wait = False
        ct = 0

    @property
    @abc.abstractmethod
    def status(self):
        return NotImplemented

    @abc.abstractmethod
    def execute(self):
        return NotImplemented
    
    @abc.abstractmethod
    def gen_ct(self):
        return NotImplemented
    
    @abc.abstractmethod
    def start_work(self):
        return NotImplemented
    
    @abc.abstractmethod
    def start_wait(self):
        return NotImplemented
    
    @abc.abstractmethod
    def release(self):
        return NotImplemented
    
class InspectStation(StationBase):

    def __init__(self, work_mean_time=0, work_std_time=0.1):
        super().__init__(self)
        self.mean_time = work_mean_time
        self.std_time = work_std_time

    ###
    def __init__(self):
        self.is_working = False
        self.is_wait = False
        self.ct = 0

    @abc.abstractmethod
    def execute(self):
        return NotImplemented
    
    @abc.abstractmethod
    def gen_ct(self):
        return NotImplemented
    
    def start_work(self):
        self.is_working = True
        return f"{self.__class__.__name__} is now working."
    
    def start_wait(self):
        self.is_working = False
        self.is_wait = True
        return f"{self.__class__.__name__} is now waiting."
    
    def release(self):
        self.is_working = False
        self.is_wait = False
        return f"{self.__class__.__name__} is released."


class MachineBase(abc.ABC):
    
    @abc.abstractmethod
    def add_station(self, station: Station):
        return NotImplemented
    
    @abc.abstractmethod
    def select_item2execute(self):
        return NotImplemented

    @abc.abstractmethod
    def select_item2execute(self):
        return NotImplemented