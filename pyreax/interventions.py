import torch, einops
from torch import nn
from pyvene import (
    IntervenableModel,
    ConstantSourceIntervention,
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
    CollectIntervention,
    InterventionOutput
)


class MaxReLUIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=True)
        # on average, some token should be initially activating the latent.
        with torch.no_grad():
            self.proj.weight.fill_(0.01)
            self.proj.bias.fill_(0)
    
    def encode(
        self, base, source=None, subspaces=None, k=1
    ):
        _weight = []
        _bias = []
        for subspace in subspaces["input_subspaces"]:
            _weight += [self.proj.weight[subspace]]
            _bias += [self.proj.bias[subspace]]
        W_c = torch.stack(_weight, dim=0).unsqueeze(dim=-1)
        b_c = torch.stack(_bias, dim=0).unsqueeze(dim=-1)

        # latent
        # base : [b, s, h]
        latent = torch.relu(torch.bmm(base, W_c).squeeze(dim=-1) + b_c)
        
        # topk over a seq
        topk_acts, topk_indices = latent.topk(k=k, dim=-1, sorted=False)

        topk_latent = torch.zeros_like(latent)
        topk_latent.scatter_(-1, topk_indices, topk_acts)
        topk_latent = topk_latent.unsqueeze(dim=-1)
        return topk_latent, latent

    def forward(
        self, base, source=None, subspaces=None
    ):
        ctrl_weight = []
        for subspace in subspaces["output_subspaces"]:
            ctrl_weight += [self.proj.weight[subspace]]
        W_ctrl = torch.stack(ctrl_weight, dim=0).unsqueeze(dim=-1).permute(0, 2, 1)
        
        topk_latent, latent = self.encode(base, source, subspaces)
        topk_latent = topk_latent.squeeze(dim=-1)
        max_latent = topk_latent.max(dim=-1, keepdim=True)[0]
        steer_dir = torch.bmm(max_latent.unsqueeze(dim=-1), W_ctrl) # bs, 1, dim
        output = base + steer_dir

        return InterventionOutput(
            output=output.to(base.dtype),
            latent=latent
        )


class SubspaceAdditionIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        # Note that we initialise these to zeros because we're loading in pre-trained weights.
        # If you want to train your own SAEs then we recommend using blah
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
                self.embed_dim, kwargs["low_rank_dimension"], bias=True)

    def forward(self, base, source=None, subspaces=None):
        steering_vec = torch.tensor(subspaces["mag"]) * self.proj.weight[subspaces["idx"]].unsqueeze(dim=0)
        output = base + steering_vec
        return output


class JumpReLUSAECollectIntervention(
    CollectIntervention
):
    """To collect SAE latent activations"""
    def __init__(self, **kwargs):
        # Note that we initialise these to zeros because we're loading in pre-trained weights.
        # If you want to train your own SAEs then we recommend using blah
        super().__init__(**kwargs, keep_last_dim=True)
        self.W_enc = nn.Parameter(torch.zeros(self.embed_dim, kwargs["low_rank_dimension"]))
        self.W_dec = nn.Parameter(torch.zeros(kwargs["low_rank_dimension"], self.embed_dim))
        self.threshold = nn.Parameter(torch.zeros(kwargs["low_rank_dimension"]))
        self.b_enc = nn.Parameter(torch.zeros(kwargs["low_rank_dimension"]))
        self.b_dec = nn.Parameter(torch.zeros(self.embed_dim))
    
    def encode(self, input_acts):
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts
    
    def forward(self, base, source=None, subspaces=None):
        acts = self.encode(base)
        return acts