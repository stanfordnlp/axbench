import torch, einops, random
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

        # Create mask for non-topk elements
        non_topk_latent = latent.clone()
        non_topk_latent.scatter_(-1, topk_indices, 0)  # Zero out topk elements

        return topk_acts, non_topk_latent, latent

    def forward(
        self, base, source=None, subspaces=None
    ):
        """h' = h + MaxReLU(h@v_c)*v_s"""

        # get steering direction
        v = []
        for subspace in subspaces["output_subspaces"]:
            v += [self.proj.weight[subspace]]
        vs = torch.stack(v, dim=0).unsqueeze(dim=-1).permute(0, 2, 1)

        # get steering magnitude
        topk_acts, non_topk_latent, latent = self.encode(
            base, source, subspaces, k=subspaces["k"])
        max_mean_latent = topk_acts.mean(dim=-1, keepdim=True)

        # steering vector
        steering_vec = torch.bmm(max_mean_latent.unsqueeze(dim=-1), vs) # bs, 1, dim

        # addition intervention
        output = base + steering_vec

        return InterventionOutput(
            output=output.to(base.dtype),
            latent=[latent, non_topk_latent]
        )


class AdditionIntervention(
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
        # use subspaces["idx"] to select the correct weight vector
        steering_vec = subspaces["max_act"].unsqueeze(dim=-1) * \
            subspaces["mag"].unsqueeze(dim=-1) * self.proj.weight[subspaces["idx"]]
        output = base + steering_vec.unsqueeze(dim=1)
        return output
    

class SubspaceAdditionIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=True)
    
    def forward(self, base, source=None, subspaces=None):
        # Get the normalized subspace vector (unit vector)
        v = self.proj.weight[subspaces["idx"]].unsqueeze(1)
        proj_coeff = (base * v).sum(dim=-1, keepdim=True)
        proj_vec = proj_coeff * v  

        steering_scale = subspaces["max_act"].unsqueeze(-1).unsqueeze(-1) * \
            subspaces["mag"].unsqueeze(-1).unsqueeze(-1)
        steering_vec = steering_scale * v
        
        # Replace the projection component with the steering vector
        output = (base - proj_vec) + steering_vec
        return output


class DictionaryAdditionIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    Anthropic's intervention method. 
    
    For smaller models, we just gave up on this ...
    But feel free to try it and see if it works for you.
    """
    def __init__(self, **kwargs):
        # Note that we initialize these to zeros because we're loading in pre-trained weights.
        # If you want to train your own SAEs then we recommend using appropriate initialization.
        super().__init__(**kwargs, keep_last_dim=True)
        self.W_enc = nn.Parameter(torch.zeros(self.embed_dim, kwargs["low_rank_dimension"]))
        self.W_dec = nn.Parameter(torch.zeros(kwargs["low_rank_dimension"], self.embed_dim))
        self.threshold = nn.Parameter(torch.zeros(kwargs["low_rank_dimension"]))
        self.b_enc = nn.Parameter(torch.zeros(kwargs["low_rank_dimension"]))
        self.b_dec = nn.Parameter(torch.zeros(self.embed_dim))
    
    def encode(self, input_acts):
        pre_acts = torch.matmul(input_acts, self.W_enc) + self.b_enc  # Shape: [batch_size, seq_len, low_rank_dimension]
        mask = (pre_acts > self.threshold)  # Shape: [batch_size, seq_len, low_rank_dimension]
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def decode(self, acts):
        reconstructed = torch.matmul(acts, self.W_dec) + self.b_dec  # Shape: [batch_size, seq_len, embed_dim]
        return reconstructed

    def forward(self, base, source=None, subspaces=None):
        """
        base: Residual stream activity x, shape [batch_size, seq_len, embed_dim]
        subspaces: Dictionary containing 'idx' and 'mag'
        """
        acts = self.encode(base)
        SAE_x = self.decode(acts)
        error_x = base - SAE_x
        
        acts_modified = acts.clone()
        feature_acts = subspaces['mag'] * subspaces["max_act"]
        acts_modified[:, :, subspaces['idx']] = feature_acts.to(base.dtype)

        modified_SAE_x = self.decode(acts_modified)
        x_new = modified_SAE_x + error_x 

        return x_new


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