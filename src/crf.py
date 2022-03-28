"""

Conditional Random Fields
Reference: https://aclanthology.org/2020.emnlp-main.516.pdf
"""

import torch
import torch.nn as nn
from dataset import IdxMaps


START_ID = 0
O_ID = 1


class CRFInference:
    """
    Inference part of the generalized CRF model
    """

    def __init__(self, n_tag, trans_priors, power):
        """
        We assume the batch size is 1, so no need to worry about PAD for now
        n_tag: START, O, and I_Xs
        """
        super().__init__()
        self.transitions = self.trans_expand(n_tag, trans_priors, power)

    @staticmethod
    def trans_expand(n_tag, priors, power):
        s_o, s_i, o_o, o_i, i_o, i_i, x_y = priors
        # self transitions for I-X tags
        a = torch.eye(n_tag) * i_i
        # transitions from I-X to I-Y
        b = torch.ones(n_tag, n_tag) * x_y / (n_tag - 3)
        c = torch.eye(n_tag) * x_y / (n_tag - 3)
        transitions = a + b - c
        # transition from START to O
        transitions[START_ID, O_ID] = s_o
        # transitions from START to I-X
        transitions[START_ID, O_ID+1:] = s_i / (n_tag - 2)
        # transition from O to O
        transitions[O_ID, O_ID] = o_o
        # transitions from O to I-X
        transitions[O_ID, O_ID+1:] = o_i / (n_tag - 2)
        # transitions from I-X to O
        transitions[O_ID+1:, O_ID] = i_o
        # no transitions to START
        transitions[:, START_ID] = 0.

        powered = torch.pow(transitions, power)
        summed = powered.sum(dim=1)

        transitions = powered / summed.view(n_tag, 1)

        transitions = torch.where(transitions > 0, transitions, torch.tensor(.000001))

        # print(transitions)
        # print(torch.sum(transitions, dim=1))
        return torch.log(transitions)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        Take the emission scores calculated by NERModel, and return a tensor of CRF features,
        which is the sum of transition scores and emission scores.
        :param scores: emission scores calculated by NERModel.
            shape: (batch_size, sentence_length, ntags)
        :return: a tensor containing the CRF features whose shape is
            (batch_size, sentence_len, ntags, ntags). F[b, t, i, j] represents
            emission[t, j] + transition[i, j] for the b'th sentence in this batch.
        """
        batch_size, sentence_len, _ = scores.size()

        # expand the transition matrix batch-wise as well as sentence-wise
        transitions = self.transitions.expand(batch_size, sentence_len, -1, -1)

        # add another dimension for the "from" state, then expand to match
        # the dimensions of the expanded transition matrix above
        emissions = scores.unsqueeze(2).expand_as(transitions)

        # add them up
        return transitions + emissions

    @staticmethod
    def viterbi(features: torch.Tensor) -> torch.Tensor:
        """
        Decode the most probable sequence of tags.
        Note that the delta values are calculated in the log space.
        :param features: the feature matrix from the forward method of CRF.
            shaped (batch_size, sentence_len, ntags, ntags)
        :return: a tensor containing the most probable sequences for the batch.
            shaped (batch_size, sentence_len)
        """
        batch_size, sentence_len, ntags, _ = features.size()

        # initialize the deltas
        delta_t = features[:, 0, START_ID, :]
        deltas = [delta_t]

        # use dynamic programming to iteratively calculate the delta values
        for t in range(1, sentence_len):
            f_t = features[:, t]
            delta_t, _ = torch.max(f_t + delta_t.unsqueeze(2).expand_as(f_t), 1)
            deltas.append(delta_t)

        # now iterate backward to figure out the most probable tags
        sequences = [torch.argmax(deltas[-1], 1, keepdim=True)]
        for t in reversed(range(sentence_len - 1)):
            f_prev = features[:, t + 1].gather(
                2, sequences[-1].unsqueeze(2).expand(batch_size, ntags, 1)).squeeze(2)
            sequences.append(torch.argmax(f_prev + deltas[t], 1, keepdim=True))
        sequences.reverse()
        return torch.cat(sequences, dim=1)


class CRF(nn.Module):
    """
    Linear Chain CRF
    """

    def __init__(self, ntags: int):
        """
        Initialize the Linear Chain CRF layer.
        :param ntags: number of tags. Usually from IdxMaps
        """
        super().__init__()
        transitions = torch.empty(ntags, ntags)
        nn.init.uniform_(transitions, -0.1, 0.1)
        # can't transition into START
        transitions[:, IdxMaps.START_ID] = -10000.0

        self.transitions = nn.Parameter(transitions)  # type: ignore

    def forward(self, scores: torch.Tensor) -> torch.Tensor:  # type: ignore
        """
        Take the emission scores calculated by NERModel, and return a tensor of CRF features,
        which is the sum of transition scores and emission scores.
        :param scores: emission scores calculated by NERModel.
            shape: (batch_size, sentence_length, ntags)
        :return: a tensor containing the CRF features whose shape is
            (batch_size, sentence_len, ntags, ntags). F[b, t, i, j] represents
            emission[t, j] + transition[i, j] for the b'th sentence in this batch.
        """
        batch_size, sentence_len, _ = scores.size()

        # expand the transition matrix batch-wise as well as sentence-wise
        transitions = self.transitions.expand(batch_size, sentence_len, -1, -1)

        # add another dimension for the "from" state, then expand to match
        # the dimensions of the expanded transition matrix above
        emissions = scores.unsqueeze(2).expand_as(transitions)

        # add them up
        return transitions + emissions

    @staticmethod
    def forward_alg(features: torch.Tensor) -> torch.Tensor:
        """
        Calculate the log alpha values using the forward algorithm.
        :param features: the features matrix from the forward method of CRF
            shaped (batch_size, sentence_len, ntags, ntags)
        :return: the tensor that represents a series of alpha values for the batch
            whose shape is (batch_size, sentence_len)
        """
        _, sentence_len, _, _ = features.size()

        # initialize the alpha value
        alpha_t = features[:, 0, IdxMaps.START_ID, :]
        alphas = [alpha_t]

        # use dynamic programming to iteratively calculate the alpha value
        for t in range(1, sentence_len):
            f_t = features[:, t]
            alpha_t = torch.logsumexp(f_t + alpha_t.unsqueeze(2).expand_as(f_t), 1)
            alphas.append(alpha_t)

        # return all the alpha values
        return torch.stack(alphas, dim=1)

    @staticmethod
    def tags_score(tags: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Calculate the score for the given sequence of tags.
        :param tags: a batch of sequences of tags whose shape is (batch_sizee, sentence_len)
        :param features: the features matrix from the forward method of CRF.
            shaped (batch_size, sentence_len, ntags, ntags)
        :return: a tensor with scores for the given sequences of tags.
            shaped (batch_size,)
        """
        batch_size, sentence_len, ntags, _ = features.size()

        # we first collect all the features whose "to" tag is given by tags,
        # i.e. F[b, t, i, *tags]
        # the resulting dimension is (batch, sentence_len, ntags, 1)
        to_idx = tags.view(-1, sentence_len, 1, 1).expand(-1, -1, ntags, -1)
        to_scores = features.gather(3, to_idx)

        # now out of to_scores, gather all the features whose "from" tag is
        # given by tags plus the start tag.
        # i.e. F[b, t, *[start + tags], j]
        # the resulting dimension is (batch, sentence_len, 1, 1)
        from_idx = torch.cat(
            (torch.tensor(IdxMaps.START_ID).expand(batch_size, 1).to(tags.device), tags[:, :-1]),
            dim=1
        )
        scores = to_scores.gather(2, from_idx.view(-1, sentence_len, 1, 1))

        # we've now gathered all the right scores, so sum them up!
        return torch.sum(scores.view(-1, sentence_len), dim=1)

    @staticmethod
    def viterbi(features: torch.Tensor) -> torch.Tensor:
        """
        Decode the most probable sequence of tags.
        Note that the delta values are calculated in the log space.
        :param features: the feature matrix from the forward method of CRF.
            shaped (batch_size, sentence_len, ntags, ntags)
        :return: a tensor containing the most probable sequences for the batch.
            shaped (batch_size, sentence_len)
        """
        batch_size, sentence_len, ntags, _ = features.size()

        # initialize the deltas
        delta_t = features[:, 0, IdxMaps.START_ID, :]
        deltas = [delta_t]

        # use dynamic programming to iteratively calculate the delta values
        for t in range(1, sentence_len):
            f_t = features[:, t]
            delta_t, _ = torch.max(f_t + delta_t.unsqueeze(2).expand_as(f_t), 1)
            deltas.append(delta_t)

        # now iterate backward to figure out the most probable tags
        sequences = [torch.argmax(deltas[-1], 1, keepdim=True)]
        for t in reversed(range(sentence_len - 1)):
            f_prev = features[:, t + 1].gather(
                2, sequences[-1].unsqueeze(2).expand(batch_size, ntags, 1)).squeeze(2)
            sequences.append(torch.argmax(f_prev + deltas[t], 1, keepdim=True))
        sequences.reverse()
        return torch.cat(sequences, dim=1)
