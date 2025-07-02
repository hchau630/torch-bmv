from torch import Tensor
import torch


def bmv(A: Tensor, x: Tensor, naive: bool = False) -> Tensor:
    """
    Optimized batched matrix-vector multiplication.

    Args:
        A: Tensor with shape (*, m, n).
        x: Tensor with shape (*, n).
        naive: If True, use the naive implementation instead of the optimized one.

    Returns:
        Tensor: Tensor with shape (*, m).
    """
    if naive:
        return (A @ x.unsqueeze(-1)).squeeze(-1)

    # einsum is much more efficient than matmul, see pytorch github issue #110858
    return torch.einsum("...ij,...j->...i", A, x)
