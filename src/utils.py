import numpy as np
import logging
import os
import torch
import random
import torch.nn.functional as F
from collections import defaultdict, Counter
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertPreTrainedModel, BertModel
from seqeval.metrics.sequence_labeling import get_entities

logger = logging.getLogger(__name__)
random.seed(0)

def nt_xent(loss, num, denom, temperature = 1):

    loss = torch.exp(loss/temperature)
    cnts = torch.sum(num, dim = 1)
    loss_num = torch.sum(loss * num, dim = 1)
    loss_denom = torch.sum(loss * denom, dim = 1)
    # sanity check
    nonzero_indexes = torch.where(cnts > 0)
    loss_num, loss_denom, cnts = loss_num[nonzero_indexes], loss_denom[nonzero_indexes], cnts[nonzero_indexes]

    loss_final = -torch.log2(loss_num) + torch.log2(loss_denom) + torch.log2(cnts)
    return loss_final

def loss_kl(mu_i, sigma_i, mu_j, sigma_j, embed_dimension):
    '''

    Calculates KL-divergence between two DIAGONAL Gaussians.
    Reference: https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians.
    Note: We calculated both directions of KL-divergence.
    '''
    sigma_ratio = sigma_j / sigma_i
    trace_fac = torch.sum(sigma_ratio, 1)
    log_det = torch.sum(torch.log(sigma_ratio + 1e-14), axis=1)
    mu_diff_sq = torch.sum((mu_i - mu_j) ** 2 / sigma_i, axis=1)
    ij_kl = 0.5 * (trace_fac + mu_diff_sq - embed_dimension - log_det)
    sigma_ratio = sigma_i / sigma_j
    trace_fac = torch.sum(sigma_ratio, 1)
    log_det = torch.sum(torch.log(sigma_ratio + 1e-14), axis=1)
    mu_diff_sq = torch.sum((mu_j - mu_i) ** 2 / sigma_j, axis=1)
    ji_kl = 0.5 * (trace_fac + mu_diff_sq - embed_dimension - log_det)
    kl_d = 0.5 * (ij_kl + ji_kl)
    return kl_d

def euclidean_distance(a, b, normalize=False):
    if normalize:
        a = F.normalize(a)
        b = F.normalize(b)
    logits = ((a - b) ** 2).sum(dim=1)
    return logits


def remove_irrelevant_tokens_for_loss(self, attention_mask, original_embedding_mu, original_embedding_sigma, labels):
    active_indices = attention_mask.view(-1) == 1
    active_indices = torch.where(active_indices == True)[0]

    output_embedding_mu = original_embedding_mu.view(-1, self.embedding_dimension)[active_indices]
    output_embedding_sigma = original_embedding_sigma.view(-1, self.embedding_dimension)[active_indices]
    labels_straightened = labels.view(-1)[active_indices]

    # remove indices with negative labels only

    nonneg_indices = torch.where(labels_straightened >= 0)[0]
    output_embedding_mu = output_embedding_mu[nonneg_indices]
    output_embedding_sigma = output_embedding_sigma[nonneg_indices]
    labels_straightened = labels_straightened[nonneg_indices]

    return output_embedding_mu, output_embedding_sigma, labels_straightened


def calculate_KL_or_euclidean(self, attention_mask, original_embedding_mu, original_embedding_sigma, labels,
                              consider_mutual_O=False, loss_type=None):

    # we will create embedding pairs in following manner
    # filtered_embedding | embedding ||| filtered_labels | labels
    # repeat_interleave |            ||| repeat_interleave |
    #                   | repeat     |||                   | repeat
    # extract only active parts that does not contain any paddings

    output_embedding_mu, output_embedding_sigma, labels_straightened = remove_irrelevant_tokens_for_loss(self, attention_mask,original_embedding_mu, original_embedding_sigma, labels)

    # remove indices with zero labels, that is "O" classes
    if not consider_mutual_O:
        filter_indices = torch.where(labels_straightened > 0)[0]
        filtered_embedding_mu = output_embedding_mu[filter_indices]
        filtered_embedding_sigma = output_embedding_sigma[filter_indices]
        filtered_labels = labels_straightened[filter_indices]
    else:
        filtered_embedding_mu = output_embedding_mu
        filtered_embedding_sigma = output_embedding_sigma
        filtered_labels = labels_straightened

    filtered_instances_nos = len(filtered_labels)

    # repeat interleave
    filtered_embedding_mu = torch.repeat_interleave(filtered_embedding_mu, len(output_embedding_mu), dim=0)
    filtered_embedding_sigma = torch.repeat_interleave(filtered_embedding_sigma, len(output_embedding_sigma),dim=0)
    filtered_labels = torch.repeat_interleave(filtered_labels, len(output_embedding_mu), dim=0)

    # only repeat
    repeated_output_embeddings_mu = output_embedding_mu.repeat(filtered_instances_nos, 1)
    repeated_output_embeddings_sigma = output_embedding_sigma.repeat(filtered_instances_nos, 1)
    repeated_labels = labels_straightened.repeat(filtered_instances_nos)

    # avoid losses with own self
    loss_mask = torch.all(filtered_embedding_mu != repeated_output_embeddings_mu, dim=-1).int()
    loss_weights = (filtered_labels == repeated_labels).int()
    loss_weights = loss_weights * loss_mask

    #ensure that the vector sizes are of filtered_instances_nos * filtered_instances_nos
    assert len(repeated_labels) == (filtered_instances_nos * filtered_instances_nos), "dimension is not of square shape."

    if loss_type == "euclidean":
        loss = -euclidean_distance(filtered_embedding_mu, repeated_output_embeddings_mu, normalize=True)

    elif loss_type == "KL":  # KL_divergence
        loss = -loss_kl(filtered_embedding_mu, filtered_embedding_sigma,
                            repeated_output_embeddings_mu, repeated_output_embeddings_sigma,
                            embed_dimension=self.embedding_dimension)

    else:
        raise Exception("unknown loss")

    # reshape the loss, loss_weight, and loss_mask
    loss = loss.view(filtered_instances_nos, filtered_instances_nos)
    loss_mask = loss_mask.view(filtered_instances_nos, filtered_instances_nos)
    loss_weights = loss_weights.view(filtered_instances_nos, filtered_instances_nos)

    loss_final = nt_xent(loss, loss_weights, loss_mask, temperature = 1)
    return torch.mean(loss_final)


class BertForTokenClassification(BertPreTrainedModel): # modified the original huggingface BertForTokenClassification to incorporate gaussian
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.embedding_dimension = config.task_specific_params['embedding_dimension']

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, self.embedding_dimension + (config.hidden_size - self.embedding_dimension) // 2)
        )

        self.output_embedder_mu = nn.Sequential(
            nn.ReLU(),
            nn.Linear(config.hidden_size,
                      self.embedding_dimension)
        )

        self.output_embedder_sigma = nn.Sequential(
            nn.ReLU(),
            nn.Linear(config.hidden_size,
                      self.embedding_dimension)
        )


        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            loss_type=None,
            consider_mutual_O=False
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = self.dropout(outputs[0])
        original_embedding_mu = ((self.output_embedder_mu((sequence_output))))
        original_embedding_sigma = (F.elu(self.output_embedder_sigma((sequence_output)))) + 1 + 1e-14
        outputs = (original_embedding_mu, original_embedding_sigma,) + (outputs[0],) + outputs[2:]

        if labels is not None:
            loss = calculate_KL_or_euclidean(self, attention_mask, original_embedding_mu,
                                                     original_embedding_sigma, labels, consider_mutual_O,
                                                     loss_type=loss_type)
            outputs = (loss,) + outputs
        return outputs  # (loss), output_mus, output_sigmas, (hidden_states), (attentions)

class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.


        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words = []
        labels = []
        for line in f:
            if line.startswith("-DOCSTART-") or line.strip() == "":
                if words:
                    examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
                    guid_index += 1
                    words = []
                    labels = []
            else:
                splits = line.split()
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1].replace("\n", ""))
                else:
                    # Examples could have no label for mode = "test"
                    labels.append("O")
        if words:
            examples.append(InputExample(guid="{}-{}".format(mode, guid_index), words=words, labels=labels))
    return examples


def convert_examples_to_features(
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        mergeB=False,
):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = defaultdict(int)
    if not mergeB:
        for i, label in enumerate(label_list):
            label_map[label] = i
    else:
        i = 0
        for label in label_list:
            if label.startswith('B-') or label.startswith('I-'):
                label_str = 'I-' + label[2:]
                if label_str not in label_map:
                    label_map[label_str] = i
                    i += 1
                label_map[label] = label_map[label_str]
            else:
                label_map[label] = i
                i += 1

    features = []
    for (ex_index, example) in enumerate(examples):

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        if sep_token_extra:
            # roberta uses an extra separator b/w pairs of sentences
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [pad_token_label_id]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [pad_token_label_id] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids
        assert len(tokens) == len(label_ids), str(tokens) + " vs" + str(label_ids)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token_label_id] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token_label_id] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length


        features.append(
            InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, label_ids=label_ids)
        )
    return features, label_map

def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    if "O" not in labels:
        labels = ["O"] + labels
    return labels


def extract_tp_actual_correct(y_true, y_pred, suffix=False):
    entities_true = defaultdict(set)
    entities_pred = defaultdict(set)
    for type_name, start, end in get_entities(y_true, suffix):
        entities_true[type_name].add((start, end))
    for type_name, start, end in get_entities(y_pred, suffix):
        entities_pred[type_name].add((start, end))

    target_names = sorted(set(entities_true.keys()) | set(entities_pred.keys()))

    tp_sum = np.array([], dtype=np.int32)
    pred_sum = np.array([], dtype=np.int32)
    true_sum = np.array([], dtype=np.int32)
    for type_name in target_names:
        entities_true_type = entities_true.get(type_name, set())
        entities_pred_type = entities_pred.get(type_name, set())
        tp_sum = np.append(tp_sum, len(entities_true_type & entities_pred_type))
        pred_sum = np.append(pred_sum, len(entities_pred_type))
        true_sum = np.append(true_sum, len(entities_true_type))

    return pred_sum, tp_sum, true_sum

def filtered_tp_counts(y_true, y_pred):
    pred_sum, tp_sum, true_sum = extract_tp_actual_correct(y_true, y_pred)
    tp_sum = tp_sum.sum()
    pred_sum = pred_sum.sum()
    true_sum = true_sum.sum()
    return pred_sum, tp_sum, true_sum
