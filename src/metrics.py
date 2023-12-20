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
    perplexity = torch.exp(cross_entropy.sum(1) / labels.shape[1])
    return perplexity


def min_k_percent_probability(
    logits: torch.Tensor,
    labels: torch.Tensor,
    k: float = 20.0,
) -> torch.Tensor:
    """Calculate Min-K % Prob in Shi et al. (2023).

    Args:
        logits (torch.Tensor): Logits of shape (batch_size, sequence_length, vocab_size).
        labels (torch.Tensor): Labels of shape (batch_size, sequence_length).
        k (float, optional): K in Min-K % Prob. Defaults to 20.0.

    Returns:
        torch.Tensor: Min-K % Prob of shape (batch_size,).

    References:
        https://arxiv.org/abs/2310.16789
    """
    ll = torch.log_softmax(logits, dim=2)  # log probabilities
    ll = ll.gather(2, labels.unsqueeze(2)).squeeze(2)  # log probabilities of labels

    num_k_percent_tokens = int(labels.shape[1] * k / 100)
    assert num_k_percent_tokens > 0, "k is too small"
    ll, _ = torch.sort(ll, dim=1)
    ll = ll[:, :num_k_percent_tokens]
    min_k_percent_probability = ll.sum(1) / num_k_percent_tokens
    return min_k_percent_probability
