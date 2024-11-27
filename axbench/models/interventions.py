import torch, random
from torch import nn
from pyvene import (
    SourcelessIntervention,
    TrainableIntervention,
    DistributedRepresentationIntervention,
    CollectIntervention,
    InterventionOutput
)


class LowRankRotateLayer(torch.nn.Module):
    """A linear transformation with orthogonal initialization."""

    def __init__(self, n, m, init_orth=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(n, m), requires_grad=True)
        if init_orth:
            torch.nn.init.orthogonal_(self.weight)

    def forward(self, x):
        return torch.matmul(x.to(self.weight.dtype), self.weight)


class TopKReLUSubspaceIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    Phi(h) = (h - h@v) + Mean(TopK(ReLU(h@v)))*v
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"])
        with torch.no_grad():
            self.proj.bias.fill_(0)

    def forward(
        self, base, source=None, subspaces=None
    ):
        v = []
        if "subspaces" in subspaces:
            for subspace in subspaces["subspaces"]:
                v += [self.proj.weight[subspace]]
        else:
            for i in range(base.shape[0]):
                v += [self.proj.weight[0]]
        v = torch.stack(v, dim=0).unsqueeze(dim=-1) # bs, h, 1
        
        # get latent
        latent = torch.relu(torch.bmm(base, v)).squeeze(dim=-1) # bs, s, 1
        topk_acts, topk_indices = latent.topk(k=subspaces["k"], dim=-1, sorted=False)
        non_topk_latent = latent.clone()
        non_topk_latent.scatter_(-1, topk_indices, 0)

        # get orthogonal component
        proj_vec = torch.bmm(latent.unsqueeze(dim=-1), v.permute(0, 2, 1)) # bs, s, 1 * bs, 1, h = bs, s, h
        base_orthogonal = base - proj_vec

        # get steering magnitude using mean of topk activations of prompt latent
        max_mean_latent = topk_acts.mean(dim=-1, keepdim=True) # bs, 1
        # steering vector
        steering_vec = max_mean_latent.unsqueeze(dim=-1) * v.permute(0, 2, 1) # bs, 1, h

        # addition intervention
        output = base_orthogonal + steering_vec

        return InterventionOutput(
            output=output.to(base.dtype),
            latent=[latent, non_topk_latent]
        )


class TopKReLUIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    Phi(h) = h + Mean(TopK(ReLU(h@v)))*v
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"])
        with torch.no_grad():
            self.proj.weight.fill_(0.01)
            self.proj.bias.fill_(0)

    def forward(
        self, base, source=None, subspaces=None
    ):
        v = []
        if "subspaces" in subspaces:
            for subspace in subspaces["subspaces"]:
                v += [self.proj.weight[subspace]]
        else:
            for i in range(base.shape[0]):
                v += [self.proj.weight[0]]
        v = torch.stack(v, dim=0).unsqueeze(dim=-1) # bs, h, 1
        
        # get latent
        latent = torch.relu(torch.bmm(base, v)).squeeze(dim=-1) # bs, s, 1
        topk_acts, topk_indices = latent.topk(k=subspaces["k"], dim=-1, sorted=False)
        non_topk_latent = latent.clone()
        non_topk_latent.scatter_(-1, topk_indices, 0)

        # get steering magnitude using mean of topk activations of prompt latent
        max_mean_latent = topk_acts.mean(dim=-1, keepdim=True) # bs, 1
        # steering vector
        steering_vec = max_mean_latent.unsqueeze(dim=-1) * v.permute(0, 2, 1) # bs, 1, h

        # addition intervention
        output = base + steering_vec

        return InterventionOutput(
            output=output.to(base.dtype),
            latent=[latent, non_topk_latent]
        )
    

class DireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    Phi(h) = h + W^T(Wh + b - Rh)
    Ref: https://arxiv.org/pdf/2404.03592
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
            kwargs["low_rank_dimension"], self.embed_dim, bias=False).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        self.learned_source = torch.nn.Linear(self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = torch.matmul(base, self.proj.weight)
        output = base + torch.matmul(
            (self.learned_source(base) - rotated_base), self.proj.weight.T
        )
        return output.to(base.dtype)
    

class LoreftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    Phi(h) = h + W^T(Wh + b - Rh)
    Ref: https://arxiv.org/pdf/2404.03592
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        proj = LowRankRotateLayer(
            self.embed_dim, kwargs["low_rank_dimension"], init_orth=True)
        self.proj = torch.nn.utils.parametrizations.orthogonal(proj)
        self.learned_source = torch.nn.Linear(self.embed_dim, kwargs["low_rank_dimension"]).to(
            kwargs["dtype"] if "dtype" in kwargs else torch.bfloat16)
        
    def forward(
        self, base, source=None, subspaces=None
    ):
        rotated_base = self.proj(base)
        output = base + torch.matmul(
            (self.learned_source(base) - rotated_base), self.proj.weight.T
        )
        return output.to(base.dtype)


class ConceptDireftIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    """
    Phi(h) = h + R^T(Wh + b - Rh)
    Ref: https://arxiv.org/pdf/2404.03592

    Note that this intervention is used for concept-based Direft.
    The main difference is that weights are assumed to be trained and saved as 3D tensors.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.W_proj = nn.Parameter(torch.zeros(
            kwargs["n_concepts"], self.embed_dim, kwargs["low_rank_dimension"]))
        self.W_source = nn.Parameter(torch.zeros(
            kwargs["n_concepts"], self.embed_dim, kwargs["low_rank_dimension"]))
        self.b_source = nn.Parameter(torch.zeros(
            kwargs["n_concepts"], kwargs["low_rank_dimension"]))

    def encode(
        self, base, source=None, subspaces=None
    ):
        """High-dimensional concept space."""
        proj_weight = self.W_proj[subspaces["input_subspaces"]] # batch_size, embed_dim, low_rank_dimension
        rotated_base = torch.bmm(base, proj_weight) # [batch_size, seq_len, embed_dim] X [batch_size, embed_dim, low_rank_dimension]

        return rotated_base # batch_size, seq_len, low_rank_dimension

    def forward(
        self, base, source=None, subspaces=None
    ):
        proj_weight = self.W_proj[subspaces["idx"]] # batch_size, embed_dim, low_rank_dimension
        source_weight = self.W_source[subspaces["idx"]] # batch_size, embed_dim, low_rank_dimension
        source_bias = self.b_source[subspaces["idx"]].unsqueeze(dim=1) # batch_size, 1, low_rank_dimension
        rotated_base = torch.bmm(base, proj_weight) # batch_size, seq_len, low_rank_dimension
        output = base + torch.bmm(
            ((torch.bmm(base, source_weight) + source_bias) - rotated_base), # batch_size, seq_len, low_rank_dimension
            proj_weight.transpose(-1, -2)
        )
        return output.to(base.dtype)
    

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


class SubspaceIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"], bias=True)
    
    def forward(self, base, source=None, subspaces=None):
        prefix_length = subspaces["prefix_length"]
        if base.shape[1] > 1:
            cached_base_prefix = base[:,:prefix_length].clone()
        v = self.proj.weight[subspaces["idx"]].unsqueeze(dim=-1) # bs, h, 1

        # get orthogonal component
        latent = torch.relu(torch.bmm(base, v)) # bs, s, 1
        proj_vec = torch.bmm(latent, v.permute(0, 2, 1)) # bs, s, 1 * bs, 1, h = bs, s, h
        base_orthogonal = base - proj_vec

        steering_scale = subspaces["max_act"].unsqueeze(-1).unsqueeze(-1) * \
            subspaces["mag"].unsqueeze(-1).unsqueeze(-1)
        steering_vec = steering_scale * v.permute(0, 2, 1) # bs, 1, h
        
        # Replace the projection component with the steering vector
        output = base_orthogonal + steering_vec 
        if base.shape[1] > 1:
            output[:,:prefix_length] = cached_base_prefix
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
    
    def forward(self, base, source=None, subspaces=None):
        pre_acts = base @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts
    

class SparseProbeIntervention(
    # We still inherit from these classes to keep it as close as possible to the LsReFT impl.
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention
):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, keep_last_dim=True)
        self.proj = torch.nn.Linear(
            self.embed_dim, kwargs["low_rank_dimension"])
        with torch.no_grad():
            self.proj.weight.fill_(0.01)
            self.proj.bias.fill_(0)

    def forward(
        self, base, source=None, subspaces=None
    ):
        v = []
        if "subspaces" in subspaces:
            for subspace in subspaces["subspaces"]:
                v += [self.proj.weight[subspace]]
        else:
            for i in range(base.shape[0]):
                v += [self.proj.weight[0]]
        v = torch.stack(v, dim=0).unsqueeze(dim=-1) # bs, h, 1
        
        # get latent
        latent = torch.relu(torch.bmm(base, v)).squeeze(dim=-1) # bs, s, 1
        topk_acts, topk_indices = latent.topk(k=subspaces["k"], dim=-1, sorted=False)
        non_topk_latent = latent.clone()
        non_topk_latent.scatter_(-1, topk_indices, 0)

        # get steering magnitude using mean of topk activations of prompt latent
        max_mean_latent = topk_acts.mean(dim=-1, keepdim=False) # bs

        return InterventionOutput(
            output=base,
            latent=[max_mean_latent, non_topk_latent, latent]
        )