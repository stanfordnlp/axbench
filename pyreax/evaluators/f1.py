from .evaluator import *

class F1Evaluator(Evaluator):
    def __init__(self, disable_tqdm=False):
        self.disable_tqdm = disable_tqdm
    
    def __str__(self):
        return 'F1Evaluator'
    
    def compute_metrics(self, examples, threshold=0.0, write_to_file=None):
        F1 = []
        for example in tqdm(examples, disable=self.disable_tqdm):
            pass
        return {
            "F1" : np.mean(F1)
        }