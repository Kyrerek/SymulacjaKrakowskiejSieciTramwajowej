import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from scipy import stats

class StopRoadSampler:
    """
        Sampler for one StopRoad modeling delay times using Log-Normal distribution.
    """

    def __init__(self,thresholds = None,dists = None):
        self.dists = dists if dists is not None else []
        self.thresholds = thresholds if thresholds is not None else []
        self.metrics = {}


    def fit(self,data: list[tuple[float,float]],dist_num: int = 5):
        """
            data - [(time_sec,delay_sec),...]
            dist_num - number of distributions (default - 5)
        """        
        sorted_data = sorted(data,key=lambda x: x[0])
        times = np.array([x[0] for x in sorted_data])
        delays = np.array([x[1] for x in sorted_data])

        self.thresholds, r2_val = self._calc_thresholds(times,delays,dist_num)
        self.dists, bucket_evals = self._calc_dists(sorted_data)

        self.metrics = {
            "tree_r2_score": r2_val,
            "buckets": bucket_evals
        }

        return self.metrics


    def predict(self,time):
        idx = 0
        for thresh in self.thresholds:
            if time < thresh:
                break
            idx += 1
        
        mean, std = self.dists[idx]
        return float(np.random.lognormal(mean=mean,sigma=std))

    def print_metrics(self):
        print("=== WYNIKI KALIBRACJI ===")
        print(f"R^2 drzewa decyzyjnego: {self.metrics['tree_r2_score']:.3f}\n")

        print("Jakość w poszczególnych okienkach czasowych:")
        for b in self.metrics['buckets']:
            print(f" Przedział {b['range']}:")
            print(f"   - Próbek: {b['count']}")
            print(f"   - Parametry: mu={b['mu']:.2f}, sigma={b['sigma']:.2f}")
            print(f"   - R^2 rozkładu (Q-Q): {b['r2_dist']:.3f}\n")



    def _calc_thresholds(self,times: np.ndarray,delays: np.ndarray,dist_num: int):
        X = times.reshape(-1,1)
        y = np.log(delays)

        tree = DecisionTreeRegressor(max_leaf_nodes=dist_num,random_state=42)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        r2 = r2_score(y,y_pred)

        raw_thresholds = tree.tree_.threshold[tree.tree_.threshold != -2]

        return sorted(raw_thresholds), float(r2)

    def _calc_dists(self,data):
        num_buckets = len(self.thresholds) + 1
        dist_data = [[] for _ in range(num_buckets)]
        
        for time,delay in data:
            idx = 0
            for thresh in self.thresholds:
                if time < thresh:
                    break
                idx += 1
            dist_data[idx].append(delay)

        dists = []
        

        bucket_evals = []
        
        bounds = [0.0] + self.thresholds + [86400.0]

        for i, bucket in enumerate(dist_data):
            start_h = bounds[i] / 3600.0
            end_h = bounds[i+1] / 3600.0
            range_str = f"{start_h:.2f}h - {end_h:.2f}h ({bounds[i]:.0f}s - {bounds[i+1]:.0f}s)"
            
            if len(bucket) < 3:
                dists.append((0.0, 0.1))
                bucket_evals.append({
                    "range": range_str, "count": len(bucket), 
                    "mu": 0.0, "sigma": 0.1, "r2_dist": 0.0
                })
                continue
                
            dist_log = np.log(bucket)
            mean = float(np.mean(dist_log))
            std = float(np.std(dist_log, ddof=1)) if len(dist_log) > 1 else 0.1
            dists.append((mean, std))

            _, fit_stats = stats.probplot(dist_log, dist="norm", fit=True)
            r2_dist = float(fit_stats[2] ** 2)
            
            bucket_evals.append({
                "range": range_str,
                "count": len(bucket),
                "mu": mean,
                "sigma": std,
                "r2_dist": r2_dist
            })

        return dists, bucket_evals