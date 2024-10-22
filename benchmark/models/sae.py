from .model import Model


class SAE(Method):
    """Empty placeholder as our SAEs are pretrained."""

    def make_model(self, **kwargs):
        pass

    def make_dataloader(self, examples, **kwargs):
        pass
    
    def train(self, examples, **kwargs):
        pass
        
    def save(self, dump_dir, **kwargs):
        pass
        
    def predict_latent(self, examples, **kwargs):
        pass

    def predict_steer(self, examples, **kwargs):
        pass

