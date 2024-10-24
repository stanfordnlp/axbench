import torch
import torch.nn as nn

class Reax(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # 投影层，将嵌入维度映射到标量
        self.proj = nn.Linear(embed_dim, 1, bias=True)
        # 初始化权重和偏置
        with torch.no_grad():
            self.proj.weight.fill_(0.01)
            self.proj.bias.fill_(0.0)

    def latent(self, base):
        """
        计算潜在激活值。
        Args:
            base (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, embed_dim]
        Returns:
            latent (torch.Tensor): 潜在激活值，形状为 [batch_size, seq_len]
        """
        # 计算 ReLU 激活后的潜在值
        latent = torch.relu(self.proj(base)).squeeze(-1)  # [batch_size, seq_len]
        return latent

    def steer(self, max_latent):
        """
        计算调整方向。
        Args:
            max_latent (torch.Tensor): 最大潜在激活值，形状为 [batch_size, 1]
        Returns:
            steer_dir (torch.Tensor): 调整方向，形状为 [batch_size, embed_dim]
        """
        # 计算调整方向
        # [b, s] * [h, 1] = [b, s, h]
        return torch.einsum("bs, h -> bsh", max_latent, self.proj.weight.squeeze(-1))

class MaxReLUIntervention(
    SourcelessIntervention,
    TrainableIntervention, 
    DistributedRepresentationIntervention):
    def __init__(self, embed_dim, low_rank_dimension):
        super().__init__()
        self.embed_dim = embed_dim
        self.low_rank_dimension = low_rank_dimension
        # 创建多个 Reax 实例，每个代表一个子空间
        self.reaxes = nn.ModuleList([Reax(embed_dim) for _ in range(low_rank_dimension)])

    def encode(self, base, source=None,subspaces=None, k=1):
        """
        对指定的输入子空间计算潜在表示。
        Args:
            base (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, embed_dim]
            subspaces (dict): 包含 "input_subspaces" 的字典，指定输入子空间的索引列表
            k (int): 选择 top-k 激活值
        Returns:
            topk_latents (torch.Tensor): top-k 激活值，形状为 [batch_size, seq_len, num_input_subspaces]
            latents (torch.Tensor): 原始激活值，形状为 [batch_size, seq_len, num_input_subspaces]
        """
        input_subspace = subspaces["input_subspaces"][0]
        # 调用 Reax 的 latent() 方法
        latent = self.reaxes[input_subspace].latent(base)  # [batch_size, seq_len]
        # 选择 top-k 激活值
        topk_acts, topk_indices = latent.topk(k=k, dim=1, sorted=False)
        topk_latent = torch.zeros_like(latent)
        topk_latent.scatter_(-1, topk_indices, topk_acts)
        return topk_latent.unsqueeze(-1), latent

    def forward(self, base, source=None,subspaces=None):
        """
        对输入进行干预。
        Args:
            base (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, embed_dim]
            subspaces (dict): 包含 "input_subspaces" 和 "output_subspaces" 的字典
            k (int): 选择 top-k 激活值
        Returns:
            output (torch.Tensor): 干预后的输出，形状为 [batch_size, seq_len, embed_dim]
            latents (torch.Tensor): 原始激活值，形状为 [batch_size, seq_len, num_input_subspaces]
        """
        input_subspaces = subspaces["input_subspaces"]
        output_subspaces = subspaces["output_subspaces"]

        # 调用 encode() 方法，获取 top-k 激活值和原始激活值
        topk_latents, latents = self.encode(base, subspaces=subspaces)

        # 计算所有输入子空间和序列位置上的最大激活值
        max_latent = topk_latents.view(topk_latents.size(0), -1).max(dim=1, keepdim=True)[0]  # [batch_size, seq_len]

        # 计算调整方向
        idx = output_subspaces[0]
        # 调用 Reax 的 steer() 方法
        steer_dir = self.reaxes[idx].steer(max_latent)  # [batch_size, embed_dim]
        # 将所有调整方向相加

        # 将调整方向应用到输入
        output = base + steer_dir # [batch_size, seq_len, embed_dim]
        return output, latents
