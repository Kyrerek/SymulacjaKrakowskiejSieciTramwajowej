from enum import Enum,auto

class TramStatus(Enum):
    SLEEP = auto(),
    ARRIVAL = auto(),
    WAIT = auto(),
    DWEL = auto(),
    LAYOVER = auto(),   # postój na pętli / oczekiwanie na odjazd z rozkładu
    TERMINATED = auto()

