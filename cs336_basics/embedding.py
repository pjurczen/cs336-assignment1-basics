import torch


class Embedding(torch.nn.Module):
    embeddings: torch.Tensor  # (num_embeddings, embedding_dim)

    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        self.embeddings = torch.nn.parameter.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(tensor=self.embeddings, mean=0, std=1, a=-3, b=3)

    # token_ids: torch.LongTensor (batch_size, sequence_length)
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings[token_ids]
