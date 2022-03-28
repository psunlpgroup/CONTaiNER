import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.nn.functional as F
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.metrics.sequence_labeling import get_entities, performance_measure
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
from crf import CRFInference


from transformers import (
    AdamW,
    BertConfig,
    BertTokenizer,
    set_seed
)
from src.utils import (
    convert_examples_to_features,
    read_examples_from_file,
    BertForTokenClassification,
    get_labels, filtered_tp_counts)

logger = logging.getLogger(__name__)

def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def finetune_support(args, model, tokenizer, labels, pad_token_label_id):
    previous_score = 1e+6 # infinity placeholder
    sup_dataset = read_and_load_examples(args, tokenizer, labels, pad_token_label_id, mode=args.support_path,
                                            mergeB=True)
    sampler = SequentialSampler(sup_dataset)
    dataloader = DataLoader(sup_dataset, sampler=sampler, batch_size=len(sup_dataset))

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate_finetuning, eps=args.adam_epsilon)
    # Train!

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    rep_index = -1

    set_seeds(args)
    while(True):
        rep_index += 1
        epoch_iterator = tqdm(dataloader, desc="Iteration", disable=True)
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            # here loss can be either KL, or euclidean.
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "token_type_ids": batch[2], "labels": batch[3],
                      "loss_type": args.finetune_loss,
                      "consider_mutual_O": args.consider_mutual_O}

            outputs = model(**inputs)
            loss = outputs[0]
            # logger.info("finetune loss at repetition "+ str(rep_index) + " : " + str(loss.item()))
            loss.backward()
            tr_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.zero_grad()

        if loss.item() > previous_score:
            # early stopping with single step patience
            break

        previous_score = loss.item()

def train(args, train_dataset, model):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.num_train_epochs > 0:
        t_total = len(train_dataloader) * args.num_train_epochs
    else:
        t_total = 0

    # Prepare optimizer and schedule (decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))

    # Train!
    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    training_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=False
    )
    set_seeds(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=False )
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "token_type_ids": batch[2], "labels": batch[3],"loss_type":args.training_loss,
                      "consider_mutual_O": args.consider_mutual_O}

            outputs = model(**inputs)
            loss = outputs[0]

            loss.backward()
            training_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.zero_grad()
            global_step += 1
                # TODO remove args.save_steps
    return global_step, training_loss / global_step if global_step > 0 else 0

def extract_target_labels(args, dataset, model):
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.eval_batch_size)
    vecs = None
    vecs_mu = None
    vecs_sigma = None
    labels = None
    model.eval()
    for batch in tqdm(dataloader, desc="Support representations"):
        batch = tuple(t.to(args.device) for t in batch)
        label_batch = batch[3]

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "token_type_ids": batch[2]}
            outputs = model(**inputs)
            output_embed_mu = outputs[0]
            output_embed_sigma = outputs[1]
            hidden_states = outputs[2]

        if vecs_mu is None:
            vecs = hidden_states.detach().cpu().numpy()
            vecs_mu = output_embed_mu.detach().cpu().numpy()
            vecs_sigma = output_embed_sigma.detach().cpu().numpy()
            labels = label_batch.detach().cpu().numpy()
        else:
            vecs = np.append(vecs, hidden_states.detach().cpu().numpy(), axis=0)
            vecs_mu = np.append(vecs_mu, output_embed_mu.detach().cpu().numpy(), axis=0)
            vecs_sigma = np.append(vecs_sigma, output_embed_sigma.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, label_batch.detach().cpu().numpy(), axis=0)
    _, _, hidden_size = vecs_mu.shape
    _, _, hidden_bert_size = vecs.shape
    vecs, vecs_mu, vecs_sigma, labels = vecs.reshape(-1, hidden_bert_size), vecs_mu.reshape(-1, hidden_size), vecs_sigma.reshape(-1, hidden_size), labels.reshape(-1)
    fil_vecs, fil_vecs_mu, fil_vecs_sigma, fil_labels = [], [], [], []
    for vec, vec_mu, vec_sigma, label in zip(vecs, vecs_mu, vecs_sigma, labels):
        if label == CrossEntropyLoss().ignore_index:
            continue
        fil_vecs.append(vec)
        fil_vecs_mu.append(vec_mu)
        fil_vecs_sigma.append(vec_sigma)
        fil_labels.append(label)
    vecs, vecs_mu, vecs_sigma, labels = torch.tensor(fil_vecs).to(args.device), torch.tensor(fil_vecs_mu).to(args.device), torch.Tensor(fil_vecs_sigma).to(args.device), torch.tensor(fil_labels).to(args.device)
    return vecs_mu.view(-1, hidden_size), vecs_sigma.view(-1, hidden_size), vecs.view(-1, hidden_bert_size), labels.view(-1)

def entitywise_max(scores, tags, addone=0, num_labels = None):
    # scores: n x m
    # tags: m
    # return: n x t
    n, m = scores.shape
    if num_labels == None:
        max_tag = torch.max(tags) + 1
    else:
        max_tag = num_labels # extra 1 is not needed since it's already 1 based counting
    ret = -100000. * torch.ones(n, max_tag+addone).to(scores.device)
    for t in range(addone, max_tag+addone):
        mask = (tags == (t-addone)).float().view(1, -1)
        masked = scores * mask
        masked = torch.where(masked < 0, masked, torch.tensor(-100000.).to(scores.device))
        ret[:, t] = torch.max(masked, dim=1)[0]
    return ret


def nearest_neighbor(args, rep_mus, rep_sigmas, rep_hidden_states, support_rep_mus, support_rep_sigmas, support_rep, support_tags, evaluation_criteria, num_labels):
    """
    Neariest neighbor decoder for the best named entity tag sequences
    """
    batch_size, sent_len, ndim = rep_mus.shape
    _, _, ndim_bert = rep_hidden_states.shape
    if evaluation_criteria == "KL":
        scores = _loss_kl(rep_mus.view(-1, ndim), rep_sigmas.view(-1,ndim), support_rep_mus, support_rep_sigmas, ndim)
        tags = support_tags[torch.argmin(scores, 1)]

    elif evaluation_criteria == "euclidean":
        scores = _euclidean_metric(rep_mus.view(-1, ndim), support_rep_mus, True)
        tags = support_tags[torch.argmax(scores, 1)]

    elif evaluation_criteria == "euclidean_hidden_state":
        scores = _euclidean_metric(rep_hidden_states.view(-1, ndim_bert), support_rep, True)
        tags = support_tags[torch.argmax(scores, 1)]

    else:
        raise Exception("Unknown decoding criteria detected. Please =specify KL/ euclidean/ euclidean_hidden_state")

    if args.temp_trans > 0:
        scores = entitywise_max(scores, support_tags, 1, num_labels)
        max_scores, tags = torch.max(scores, 1)
        tags = tags - 1

    return tags.view(batch_size, sent_len), scores.view(batch_size, sent_len, -1)

def _euclidean_metric(a, b, normalize=False):
    if normalize:
        a = F.normalize(a)
        b = F.normalize(b)
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b) ** 2).sum(dim=2)
    return logits

def _loss_kl(mu_i, sigma_i, mu_j, sigma_j, embed_dimension):
    n = mu_i.shape[0]
    m = mu_j.shape[0]

    mu_i = mu_i.unsqueeze(1).expand(n,m, -1)
    sigma_i = sigma_i.unsqueeze(1).expand(n,m,-1)
    mu_j = mu_j.unsqueeze(0).expand(n,m,-1)
    sigma_j = sigma_j.unsqueeze(0).expand(n,m,-1)
    sigma_ratio = sigma_j / sigma_i
    trace_fac = torch.sum(sigma_ratio, 2)
    log_det = torch.sum(torch.log(sigma_ratio + 1e-14), axis=2)
    mu_diff_sq = torch.sum((mu_i - mu_j) ** 2 / sigma_i, axis=2)
    ij_kl = 0.5 * (trace_fac + mu_diff_sq - embed_dimension - log_det)
    sigma_ratio = sigma_i / sigma_j
    trace_fac = torch.sum(sigma_ratio, 2)
    log_det = torch.sum(torch.log(sigma_ratio + 1e-14), axis=2)
    mu_diff_sq = torch.sum((mu_j - mu_i) ** 2 / sigma_j, axis=2)
    ji_kl = 0.5 * (trace_fac + mu_diff_sq - embed_dimension - log_det)
    kl_d = 0.5 * (ij_kl + ji_kl)
    return kl_d


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    sup_dataset = read_and_load_examples(args, tokenizer, labels, pad_token_label_id, mode=args.support_path, mergeB=True)
    sup_mus, sup_sigmas, sups, sup_labels = extract_target_labels(args, sup_dataset, model)
    eval_dataset = read_and_load_examples(args, tokenizer, labels, pad_token_label_id, mode=mode, mergeB=True)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    preds = None
    out_label_ids = None

    model.eval()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)
        label_batch = batch[3]

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "token_type_ids": batch[2]}
            outputs = model(**inputs)
            hidden_states = outputs[2]
            output_embedding_mu = outputs[0]
            output_embedding_sigma = outputs[1]

            nn_predictions, nn_scores = nearest_neighbor(args, output_embedding_mu, output_embedding_sigma, hidden_states, sup_mus, sup_sigmas, sups, sup_labels, evaluation_criteria=args.evaluation_criteria, num_labels=len(labels))
        if preds is None:
            preds = nn_predictions.detach().cpu().numpy()
            scores = nn_scores.detach().cpu().numpy()
            out_label_ids = label_batch.detach().cpu().numpy()

        else:
            preds = np.append(preds, nn_predictions.detach().cpu().numpy(), axis=0)
            scores = np.append(scores, nn_scores.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, label_batch.detach().cpu().numpy(), axis=0)

    merged_labels = [label for label in labels if not label.startswith('I-')]
    conv_labels = []
    for label in merged_labels:
        if label.startswith('B-'):
            conv_labels.append('I-' + label[2:])
        else:
            conv_labels.append(label)
    label_map = {i: label for i, label in enumerate(conv_labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    scores_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                scores_list[i].append(scores[i][j])
                preds_list[i].append(label_map[preds[i][j]])

    if args.temp_trans > 0:
        # START: Viterbi!!!
        vit_preds_list = [[] for _ in range(out_label_ids.shape[0])]
        crf = CRFInference(len(label_map) + 1, args.trans_priors, args.temp_trans)
        for i in range(out_label_ids.shape[0]):
            sent_scores = torch.tensor(scores_list[i])
            sent_probs = F.softmax(sent_scores, dim=1)
            sent_len, n_tag = sent_probs.shape
            feats = crf.forward(torch.log(sent_probs).view(1, sent_len, n_tag))
            vit_tags = crf.viterbi(feats)
            vit_tags = vit_tags.view(sent_len)
            vit_tags = vit_tags.detach().cpu().numpy()
            for tag in vit_tags:
                vit_preds_list[i].append(label_map[tag - 1])
        preds_list = vit_preds_list
        # END

    performance_dict = performance_measure(out_label_list, preds_list)
    pred_sum, tp_sum, true_sum = filtered_tp_counts(out_label_list, preds_list)
    results = {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
        "TP": performance_dict['TP'],
        "TN": performance_dict['TN'],
        "FP": performance_dict['FP'],
        "FN": performance_dict['FN'],
        "pred_sum": pred_sum,
        "tp_sum": tp_sum,
        "true_sum": true_sum
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list


def read_and_load_examples(args, tokenizer, labels, pad_token_label_id, mode, mergeB=False):
    examples = read_examples_from_file(args.data_dir, mode)
    features, label_map = convert_examples_to_features(
        examples,
        labels,
        args.max_seq_length,
        tokenizer,
        cls_token_at_end=False,
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=0,
        pad_token_label_id
        =pad_token_label_id,
        mergeB=mergeB,
    )

    # Convert to Tensors
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    return dataset

def trans_stats(args, labels):
    '''

    Reference: https://aclanthology.org/2020.emnlp-main.516.pdf
    '''
    tag_lists = get_tags(args.data_dir + '/train.txt', labels, True)
    s_o, s_i = 0., 0.
    o_o, o_i = 0., 0.
    i_o, i_i, x_y = 0., 0., 0.
    for tags in tag_lists:
        if tags[0] == 'O': s_o += 1
        else: s_i += 1
        for i in range(len(tags)-1):
            p, n = tags[i], tags[i+1]
            if p == 'O':
                if n == 'O': o_o += 1
                else: o_i += 1
            else:
                if n == 'O':
                    i_o += 1
                elif p != n:
                    x_y += 1
                else:
                    i_i += 1
    ret = []
    ret.append(s_o / (s_o + s_i))
    ret.append(s_i / (s_o + s_i))
    ret.append(o_o / (o_o + o_i))
    ret.append(o_i / (o_o + o_i))
    ret.append(i_o / (i_o + i_i + x_y))
    ret.append(i_i / (i_o + i_i + x_y))
    ret.append(x_y / (i_o + i_i + x_y))

    return ret


def get_tags(fname, labels, to_I=False):
    tag_lists = []
    tag_list = []
    with open(fname) as f:
        for line in f:
            if line.startswith("-DOCSTART-") or line.strip() == "":
                if tag_list:
                    tag_lists.append(tag_list)
                    tag_list = []
            else:
                splits = line.split()
                if len(splits) > 1:
                    tag = splits[1]
                    if tag not in labels:
                        tag = 'O'
                    if to_I and tag.startswith('B-'):
                        tag = 'I-' + tag[2:]
                    tag_list.append(tag)
        if tag_list:
            tag_lists.append(tag_list)

    return tag_lists


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the few_nerd task.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions will be written.",
    )
    parser.add_argument(
        "--saved_model_dir",
        default=None,
        type=str,
        required=False,
        help="The output directory where the model will be written.",
    )

    parser.add_argument(
        "--support_path",
        default=None,
        type=str,
        required=True,
        help="The file path for the support set.",
    )
    
    parser.add_argument(
        "--test_path",
        default=None,
        type=str,
        required=True,
        help="The file path for the test set.",
    )

    # Other parameters
    parser.add_argument(
        "--labels-train",
        default="",
        type=str,
        help="Path to a file containing all train labels.",
    )
    parser.add_argument(
        "--labels-test",
        default="",
        type=str,
        help="Path to a file containing all test labels.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")

    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--eval_batch_size", default=16, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1, type=float, help="Total number of training epochs to perform."
    )

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--n_classes", default=6, type=int, help="number of classes")
    parser.add_argument("--n_shots", default=5, type=int, help="number of shots.")
    parser.add_argument("--embedding_dimension", default=32, type=int, help="dimension of output embedding")
    parser.add_argument("--do_finetune_support_only", default=True, type=bool, help="Whether to finetune the model on the support set.")
    parser.add_argument("--training_loss", type=str, default="KL", help="What type of loss to use, KL, euclidean, or joint of KL and classification")
    parser.add_argument("--finetune_loss", type=str, default="KL", help= "What type of loss to use, KL, euclidean, or joint of KL and classification")
    parser.add_argument("--evaluation_criteria", type=str, default="euclidean", help= "What type of loss to use, KL, euclidean, or euclidean_hidden_state")
    parser.add_argument("--consider_mutual_O", action="store_true", help= "Do you want consider the distances of all -O tokens with all the other -O tokens too?.")
    parser.add_argument("--learning_rate_finetuning", default=5e-5, type=float, help="The initial learning rate for Adam during finetune.")
    parser.add_argument("--select_gpu", type=int, default=0, help="select on which gpu to train.")
    parser.add_argument("--silent", action="store_true", help="whether to output all INFO in training and finetuning")
    parser.add_argument("--temp_trans", default=-1, type=float, help="transition re-normalizing temperature")

    args = parser.parse_args()

    if (
        os.path.exists(args.saved_model_dir)
        and os.listdir(args.saved_model_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to override.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda:" + str(args.select_gpu) if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    args.n_gpu = min(1, args.n_gpu) # we are keeping ourselves restricted to only 1 gpu

    args.device = device
    args.best_validation_f1 = -1

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.warning(
        "Device: %s, n_gpu: %s",
        device,
        args.n_gpu,
    )

    # Set seed
    set_seeds(args)
    labels_train = get_labels(args.labels_train)
    labels_test = get_labels(args.labels_test)
    num_labels = len(labels_train)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    config = BertConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        id2label={str(i): label for i, label in enumerate(labels_train)},
        label2id={label: i for i, label in enumerate(labels_train)},
        cache_dir=args.cache_dir if args.cache_dir else None,
        task_specific_params={"embedding_dimension": args.embedding_dimension}
    )
    TOKENIZER_ARGS = ["do_lower_case", "strip_accents", "keep_accents", "use_fast"]

    tokenizer_args = {k: v for k, v in vars(args).items() if v is not None and k in TOKENIZER_ARGS}
    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        **tokenizer_args,
    )
    model = BertForTokenClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None
    )

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = read_and_load_examples(args, tokenizer, labels_train, pad_token_label_id, mode="train", mergeB=True)
        global_step, tr_loss = train(args, train_dataset, model)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        if args.saved_model_dir is not None:
            if not os.path.exists(args.saved_model_dir):
                os.makedirs(args.saved_model_dir)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint")

        model_to_save = (
            model.module if hasattr(model, "module") else model
        )
        if args.saved_model_dir is None:
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        else:
            model_to_save.save_pretrained(args.saved_model_dir)
            tokenizer.save_pretrained(args.saved_model_dir)
            torch.save(args, os.path.join(args.saved_model_dir, "training_args.bin"))

    # Evaluation
    results = {}

    set_seeds(args)
    if args.do_finetune_support_only:
        if args.silent == True:
            logging.disable(logging.ERROR)
        if args.saved_model_dir is None:
            tokenizer = BertTokenizer.from_pretrained(args.output_dir, **tokenizer_args)
            model = BertForTokenClassification.from_pretrained(args.output_dir)
        else:
            tokenizer = BertTokenizer.from_pretrained(args.saved_model_dir, **tokenizer_args)
            model = BertForTokenClassification.from_pretrained(args.saved_model_dir)
        model.to(args.device)

        finetune_support(args, model, tokenizer, labels_test, pad_token_label_id)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    if args.do_predict:
        if args.temp_trans > 0:
            args.trans_priors = trans_stats(args, labels_train)
        result, predictions = evaluate(args, model, tokenizer, labels_test, pad_token_label_id, mode=args.test_path)
        sys.stdout.write(str(result["f1"]))
        sys.stdout.flush()
        # Save results
        output_test_results_file = os.path.join(args.output_dir, "results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("{} = {}\n".format(key, str(result[key])))
        # Save predictions
        output_test_predictions_file = os.path.join(args.output_dir, "predictions.txt")
        with open(output_test_predictions_file, "w") as writer:
            with open(os.path.join(args.data_dir, "{}.txt".format(args.test_path)), "r") as f:
                example_id = 0
                prev_null = False
                for line in f:
                    if line.startswith("-DOCSTART-") or line.strip() == "":
                        writer.write(line)
                        if not prev_null and not predictions[example_id]:
                            example_id += 1
                        prev_null = True
                    elif predictions[example_id]:
                        output_line = line.split()[0] + " " + predictions[example_id].pop(0) + "\n"
                        writer.write(output_line)
                        prev_null = False
                    else:
                        logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])
                        output_line = line.split()[0] + " O\n"
                        writer.write(output_line)
                        prev_null = False

    return results


if __name__ == "__main__":
    main()