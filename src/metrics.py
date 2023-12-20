import torch
from torch.nn import CrossEntropyLoss


def perplexity(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Calculate perplexity for each sequence.

    Args:
        logits (torch.Tensor): Logits of shape (batch_size, sequence_length, vocab_size).
        labels (torch.Tensor): Labels of shape (batch_size, sequence_length).

    Returns:
        torch.Tensor: Perplexity of shape (batch_size,).
    """
    loss_fct = CrossEntropyLoss(reduction="none")
    cross_entropy = loss_fct(logits.transpose(1, 2), labels)
    return torch.exp(cross_entropy.sum(1) / labels.shape[1])
