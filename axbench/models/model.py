from abc import ABC, abstractmethod


class Model(ABC):

    def make_model(self, **kwargs):
        pass

    def make_dataloader(self, examples, **kwargs):
        pass
    
    @abstractmethod
    def train(self, examples, **kwargs):
        pass
        
    def save(self, dump_dir, **kwargs):
        pass

    def load(self, dump_dir=None, **kwargs):
        pass
    
    def predict_latent(self, examples, **kwargs):
        pass

    def predict_steer(self, examples, **kwargs):
        pass

