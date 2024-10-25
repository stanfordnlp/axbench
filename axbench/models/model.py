from abc import ABC, abstractmethod
import torch

class Model(ABC):

    def make_model(self, **kwargs):
        pass

    def make_dataloader(self, examples, **kwargs):
        pass
    
    @abstractmethod
    def train(self, examples, **kwargs):
        pass
        
    def save(self, dump_dir, **kwargs):
        weight_file = dump_dir / f"{self.__str__()}_weight.pt"
        weight = self.linear_model.proj.weight.data.cpu()
        if weight_file.exists():
            weight = torch.cat([torch.load(weight_file), weight], dim=0)
        torch.save(weight, weight_file)
        
        bias_file = dump_dir / f"{self.__str__()}_bias.pt"
        bias = self.linear_model.proj.bias.data.cpu()
        if bias_file.exists():
            bias = torch.cat([torch.load(bias_file), bias], dim=0)
        torch.save(bias, bias_file)

    def load(self, dump_dir=None, **kwargs):
        weight = torch.load(
            f"{dump_dir}/{self.__str__()}_weight.pt"
        )
        bias = torch.load(
            f"{dump_dir}/{self.__str__()}_bias.pt"
        )
        self.linear_model.proj.weight.data = weight.cuda()
        self.linear_model.proj.bias.data = bias.cuda()
    
    def predict_latent(self, examples, **kwargs):
        pass

    def predict_steer(self, examples, **kwargs):
        pass

