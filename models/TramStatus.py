from enum import Enum,auto

class TramStatus(Enum):
    ARRIVAL = auto(),
    WAIT = auto(),
    DWEL = auto(),
    TERMINATED = auto()
