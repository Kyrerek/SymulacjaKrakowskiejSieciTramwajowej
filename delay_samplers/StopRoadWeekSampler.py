from .StopRoadSampler import StopRoadSampler
import numpy as np
import datetime

class StopRoadWeekSampler:
    """
            Sampler for one StopRoad modeling delay times using Log-Normal distribution.
        """
    
    def __init__(self,samplers: list[StopRoadSampler] = None):
        self.samplers = samplers if samplers is not None else []


    def fit(self,week_data: list[list[tuple[float,float]]],dist_nums: list[int] | int = 5):
        """
            data - [(time_sec,delay_sec),...]
            dist_num - number of distributions (default - 5)
        """

        if isinstance(dist_nums,int):
            dist_nums = [dist_nums] * len(week_data)

        self.samplers = []
        metrics = []
        for data, dist_num in zip(week_data,dist_nums):
            sampler = StopRoadSampler()
            sampler.fit(data,dist_num)        
            self.samplers.append(sampler)
            metrics.append(sampler.metrics)
        return metrics

    def predict(self,time: datetime):
        week_day = time.weekday()
        if week_day < 0 or week_day >= len(self.samplers):
            raise Exception(f"Invalid day of the week: {week_day}")

        return self.samplers[week_day].predict(time)