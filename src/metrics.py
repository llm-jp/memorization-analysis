import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from torch.nn import CrossEntropyLoss


def perplexity(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Calculate perplexity for each sequence.

    Args:
        logits (torch.Tensor): Logits of shape (batch_size, sequence_length, vocab_size).
        labels (torch.Tensor): Labels of shape (batch_size, sequence_length).

    Returns:
        torch.Tensor: Perplexity of shape (batch_size,).
    """
    batch_size, sequence_length, vocab_size = logits.shape
    assert labels.shape == (batch_size, sequence_length)

    loss_fct = CrossEntropyLoss(reduction="none")
    cross_entropy = loss_fct(logits.transpose(1, 2), labels)
    perplexity_ = torch.exp(cross_entropy.sum(1) / labels.shape[1])
    return perplexity_


def min_k_percent_prob(
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
    batch_size, sequence_length, vocab_size = logits.shape
    assert labels.shape == (batch_size, sequence_length)

    ll = torch.log_softmax(logits, dim=2)  # log probabilities
    ll = ll.gather(2, labels.unsqueeze(2)).squeeze(2)  # log probabilities of labels

    num_k_percent_tokens = int(labels.shape[1] * k / 100)
    assert num_k_percent_tokens > 0, "k is too small"
    ll, _ = torch.sort(ll, dim=1)
    ll = ll[:, :num_k_percent_tokens]
    min_k_percent_prob_ = ll.sum(1) / num_k_percent_tokens
    return min_k_percent_prob_


def extractable(
    output_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Calculate Extractable in Carlini et al. (2023).

    Args:
        output_ids (torch.Tensor): Output IDs of shape (batch_size, sequence_length).
        labels (torch.Tensor): Labels of shape (batch_size, sequence_length).

    Returns:
        torch.Tensor: Extractable of shape (batch_size,).

    References:
        https://openreview.net/forum?id=TatRHT_1cK
    """
    batch_size, sequence_length = output_ids.shape
    assert labels.shape == (batch_size, sequence_length)

    extractable_ = (output_ids == labels).sum(1) == labels.shape[1]
    return extractable_


def bleu(
    output_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Calculate BLEU score.

    Args:
        output_ids (torch.Tensor): Output IDs of shape (batch_size, sequence_length).
        labels (torch.Tensor): Labels of shape (batch_size, sequence_length).

    Returns:
        torch.Tensor: BLEU score of shape (batch_size,).
    """
    batch_size, sequence_length = output_ids.shape
    assert labels.shape == (batch_size, sequence_length)

    bleu_ = []
    for i in range(batch_size):
        hypothesis = output_ids[i].tolist()
        references = [labels[i].tolist()]
        chencherry = SmoothingFunction()
        bleu_.append(sentence_bleu(references, hypothesis, smoothing_function=chencherry.method1))  # noqa
    return torch.tensor(bleu_)
