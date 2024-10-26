import torch


class Model(object):

    def __init__(self, model, tokenizer, layer, training_args=None, **kwargs):
        self.model = model
        self.tokenizer = tokenizer
        # abstracting layer
        self.layer = layer
        self.training_args = training_args

    def make_model(self, **kwargs):
        pass

    def make_dataloader(self, examples, **kwargs):
        pass
    
    def train(self, examples, **kwargs):
        pass
        
    def save(self, dump_dir, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        weight_file = dump_dir / f"{model_name}_weight.pt"
        weight = self.ax.proj.weight.data.cpu()
        if weight_file.exists():
            weight = torch.cat([torch.load(weight_file), weight], dim=0)
        torch.save(weight, weight_file)
        
        bias_file = dump_dir / f"{model_name}_bias.pt"
        bias = self.ax.proj.bias.data.cpu()
        if bias_file.exists():
            bias = torch.cat([torch.load(bias_file), bias], dim=0)
        torch.save(bias, bias_file)

    def load(self, dump_dir=None, **kwargs):
        model_name = kwargs.get("model_name", self.__str__())
        weight = torch.load(
            f"{dump_dir}/{model_name}_weight.pt"
        )
        bias = torch.load(
            f"{dump_dir}/{model_name}_bias.pt"
        )
        self.make_model(low_rank_dimension=weight.shape[1], **kwargs)
        self.ax.proj.weight.data = weight.cuda()
        self.ax.proj.bias.data = bias.cuda()
    
    def predict_latent(self, examples, **kwargs):
        pass

    def predict_steer(self, examples, **kwargs):
        pass

