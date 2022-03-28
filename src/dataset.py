import torch
import torch.nn as nn

from typing import Generator, TextIO, Optional, List, TypeVar, Set, Dict, Tuple
from collections import namedtuple

from embedding import GloveEmbedding


TagSpan = namedtuple('TagSpan', ['content', 'start', 'end'])

# Record the important things about a tag and its position in a list
TagPos = namedtuple('TagPos', ['prefix', 'content', 'pos'])

Sentence = List[str]
Tags = List[str]
T = TypeVar('T')


class TaggedSentence:
    """
    Class that represents a sentence in a BIO dataset.
    """
    DOCSTART = '-DOCSTART-'

    def __init__(self):
        self.tokens = []
        self.tags = []

    def append(self, token: str, tag: str):
        """
        Append a token with its tag. If it's the token is DOCSTART,
        we either skip it if it's in the beginning, or if there are other tokens,
        i.e. the DOCSTART is in the middle of the sentence, it raises a RuntimeError.
        :param token: token to be appended
        :param tag: the corresponding tag to be appended
        :raises: RuntimeError: A DOCSTART token was found in the middle of the sentence.
        """
        if token == self.DOCSTART:
            if self:
                raise RuntimeError(f'{self.DOCSTART} in middle of sentence')
            return
        self.tokens.append(token)
        self.tags.append(tag)

    def __len__(self):
        return len(self.tokens)


class TagSequenceParser:
    """
    Parse the BIO tagged sequences.
    Given a sequence that looks like ['B-LOC', 'I-LOC', 'O'],
    we want to ignore the "outside" tag ('O'), and calculate the spans for other tags.
    In the example, it would parse out ('LOC', 0, 2).
    """
    # these are special prefixes in various BIO formats
    START_PREFIXES = ['B', 'U', 'S']

    def __init__(self):
        """
        Initialize a TagSequenceParser
        """
        self.seq = [self._to_tagpos(-1, IdxMaps.OUT)]

    def on_tag(self, i: int, tag: str) -> Optional[TagSpan]:
        """
        Process a tag at the given position. Optionally return a TagSpan,
        if the current sequence of tags has ended, i.e. the given tag is an
        "outside" tag.
        :param i: the position of the tag
        :param tag: the tag to be processed.
        :return: Optional TagSpan
        """
        next_tagpos = self._to_tagpos(i, tag)
        if next_tagpos.prefix in self.START_PREFIXES:
            return self._reset(next_tagpos)
        else:
            if self.seq[-1].content == next_tagpos.content:
                self.seq.append(next_tagpos)
                return None
            else:
                return self._reset(next_tagpos)

    def _reset(self, next_tagpos: TagPos) -> Optional[TagSpan]:
        """
        Resets the current tag sequence.
        :param next_tagpos: the next TagPos to reset the current sequence with.
        :return: Optional completed TagSpan
        """
        ret = TagSpan(self.seq[0].content, self.seq[0].pos, next_tagpos.pos)
        self.seq = [next_tagpos]
        if ret.content not in (IdxMaps.PAD, IdxMaps.OUT, IdxMaps.START):
            return ret
        return None

    @staticmethod
    def _to_tagpos(i: int, tag: str) -> TagPos:
        """
        Pare the tag to into TagPos
        :param i: position of the tag
        :param tag: tag to be parsed.
        :return: a TagPos with the tag prefix, content and position
        """
        if tag in (IdxMaps.PAD, IdxMaps.OUT, IdxMaps.START):
            return TagPos(None, tag, i)
        prefix, content = tag.split('-', 1)
        return TagPos(prefix, content, i)

    @staticmethod
    def parse(tags: List[str]) -> Generator[TagSpan, None, None]:
        """
        Parse the given list of tags, and return a generator that yields
        TagSpans for all the detected entities.
        :param tags: a list of tags
        :return: a generator that yields TagSpans
        """
        parser = TagSequenceParser()
        for i, tag in enumerate(tags + [IdxMaps.OUT]):
            tagspan = parser.on_tag(i, tag)
            if tagspan is not None:
                yield tagspan


class BioParser:
    """
    Parser for BIO files that yields tagged sentences.
    """

    def __init__(self, fileh: TextIO):
        """
        Initialize a BioParser with the given BIO file.
        :param fileh: handler for the BIO file.
        """
        self.fileh = fileh

    def rewind(self):
        """
        Go back to the start of the file.
        """
        self.fileh.seek(0)

    def __iter__(self) -> Generator[TaggedSentence, None, None]:
        """
        Return a generator that yields TaggedSentences that are not empty.
        :return: a generator that yields parsed TaggedSentences from the doc.
        """
        yield from filter(None, self.get_tagged_sentences())

    def get_tagged_sentences(self) -> Generator[TaggedSentence, None, None]:
        """
        Split the file into sections delimited by empty lines,
        then transform each section as a TaggedSentence and yield it.
        :return: a generator that yields a TaggedSentence, which could be empty.
        """
        tagged_sentence = TaggedSentence()
        for line in self.fileh:
            line = line.strip()
            if not line:
                yield tagged_sentence
                tagged_sentence = TaggedSentence()
            else:
                token, tag = line.split()
                tagged_sentence.append(token, tag)
        yield tagged_sentence


class IdxMaps:
    """
    The models don't work on the raw strings but integers that represent strings (and chars)
    The training will first make an inventory of all the tokens that have been seen in the
    data sets.
    In this class we will implement functions that convert a sentence (list of tokens)
    to a list of integers. And also convert a sentence to a matrix of integers,
    where each row represnts a token in the sentence and the columns are the chars of the tokens
    mapped to integers
    """
    PAD_ID = 0
    START_ID = 1

    UNK_ID = 1

    PAD = ''
    START = '<start>'
    OUT = 'O'

    def __init__(self, tokens: Set[str], tags: Set[str]):
        """
        For charcter IDs and token IDs, we start from 2 as 0 and 1 correspond to PAD and UNK.
        For tag IDs, we start from 2 as 0 and 1 correspond to PAD and START.
        :param tokens: unique tokens that will be seen in training
        :param tags: unique tags that are present in the dataset
        """
        # 0 and 1 are reserved for PAD and UNK characters, so we will start at 2
        # for tags there is no notion of UNK
        # UNK will only play a role during prediction time when we might see
        # a token/char that was not present in training
        lowercase_tokens = sorted(set(token.lower() for token in tokens))
        self.token_idx_map: Dict[str, int] = {
            token: (i + 2) for i, token in enumerate(lowercase_tokens)}
        self.token_idx_map[self.PAD] = self.PAD_ID
        chars = self.get_chars(tokens)
        self.char_idx_map: Dict[str, int] = {c: (i + 2) for i, c in enumerate(chars)}
        self.tag_idx_map: Dict[str, int] = {tag: (i + 2) for i, tag in enumerate(sorted(tags))}
        self.active_classes: Set[str] = {tag[2:] for tag in tags if tag.startswith('B-')}
        self.tag_idx_map.update({
            self.PAD: self.PAD_ID,
            self.START: self.START_ID,
        })
        sorted_tags = sorted((idx, tag) for tag, idx in self.tag_idx_map.items())
        self.idx_tag_map = [t for _, t in sorted_tags]

    @classmethod
    def get_tokens_and_tags(cls, *datasets: List[TaggedSentence]) -> Tuple[Set[str], Set[str]]:
        tokens: Set[str] = set()
        tags: Set[str] = set()
        for dataset in datasets:
            for tagged_sentence in dataset:
                tokens.update(tagged_sentence.tokens)
                tags.update(tagged_sentence.tags)
        return tokens, tags

    @classmethod
    def from_datasets(cls, *datasets: List[TaggedSentence]):
        """
        Create an IdxMaps from tagged sentences.
        :param datasets: a list of tagged sentences
        """
        return cls(*cls.get_tokens_and_tags(*datasets))

    def tag_idx_remap(self, active_classes: List[str], collapseB: bool):
        """
        Recreate tag_idx_map and idx_tag_map based on active classes.
        This method is mainly useful for few-shot NER setting.
        :param active_classes: entity classes of interests
        :param collapseB: whether to map 'B-' to 'I-' or not
        """
        self.active_classes = set(active_classes)
        new_idx_tag_map: List[str] = []
        for tag in self.idx_tag_map:
            if tag.startswith('B-') or tag.startswith('I-'):
                pre, X = tag.split('-')
                if X in self.active_classes:
                    if pre == 'B':
                        if not collapseB:
                            new_idx_tag_map.append(tag)
                    else:
                        new_idx_tag_map.append(tag)
            else:
                new_idx_tag_map.append(tag)
        self.idx_tag_map = new_idx_tag_map
        temp_tag_idx_map = {tag: idx for idx, tag in enumerate(new_idx_tag_map)}
        for tag in self.tag_idx_map:
            if tag not in temp_tag_idx_map:
                if tag.startswith('B-') and ('I-' + tag[2:] in temp_tag_idx_map):
                    self.tag_idx_map[tag] = temp_tag_idx_map['I-' + tag[2:]]
                else:
                    self.tag_idx_map[tag] = temp_tag_idx_map['O']
            else:
                self.tag_idx_map[tag] = temp_tag_idx_map[tag]

    def get_active_tags(self, tags: Tags) -> Tags:
        """Remap tags of non-active classes to 'O'
        """
        ret_tags = []
        for tag in tags:
            if tag.startswith('B-') or tag.startswith('I-'):
                X = tag[2:]
                if X in self.active_classes:
                    ret_tags.append(tag)
                else:
                    ret_tags.append('O')
            else:
                ret_tags.append(tag)
        return ret_tags

    def num_chars(self) -> int:
        """
        Return the number of characters including PAD and UNK
        :return: number of characters with PAD and UNK
        """
        return len(self.char_idx_map) + 2

    def num_tags(self) -> int:
        """
        Return the number of tags including PAD and START
        :return: number of tags with PAD and START
        """
        return len(self.idx_tag_map)

    def to_state_dict(self) -> Dict:
        """
        Dict representation of the class. Used for serialization.
        :return: Dict representation
        """
        state_dict = dict(self.__dict__)
        state_dict.pop('_schema', None)
        state_dict.pop('_saved_kwargs', None)
        state_dict.pop('_extensions', None)
        state_dict.pop('_created_with_tag', None)
        return state_dict

    def load_state_dict(self, state_dict: Dict):
        """
        Load the given state dictionary.
        :param state_dict: a state_dict
        """
        for attr, val in state_dict.items():
            setattr(self, attr, val)

    @classmethod
    def from_state_dict(cls, state_dict: Dict):
        """
        Inverse of to_state_dict. Used for deserialization.
        :param state_dict: a state_dict
        """
        instance = cls(set(), set())
        instance.load_state_dict(state_dict)
        return instance

    def form_pretrained_token_embedding(self, glove_embedding: GloveEmbedding) -> nn.Embedding:
        """
        Extract the glove embeddings for the tokens.
        Note that we will prepend a zero vector and a random vector to represent PAD and UNK.
        As a result, the returned array is one longer than sorted(self.token_idx_map).
        :param glove_embedding: glove embeddings for the tokens
        :return: pytorch Embedding with glove embeddings for the tokens
        """
        vecs = [glove_embedding.zeros(), glove_embedding.random()]  # arbitraray random for unk
        vecs.extend(glove_embedding.get(token) for token in sorted(self.token_idx_map) if token)
        return nn.Embedding.from_pretrained(torch.tensor(vecs, dtype=torch.float), freeze=False)

    @staticmethod
    def get_chars(tokens: Set[str]) -> List[str]:
        """
        Extract a unique, sorted list of characters from the given set of tokens.
        :param tokens: a set of tokens from which the list of characters will be extracted
        :return: a unique, sorted list of characters
        """
        chars: Set[str] = set()
        for token in tokens:
            chars.update(token)
        return sorted(chars)

    def sentence_to_idxlist(self, sentence: Sentence) -> List[int]:
        """
        Transform the given sentence (list of string tokens) into a list of token IDs.
        :param sentence: sentence to be transformed into a list of integer token IDs.
        :return: a list of integer token IDs.
        """
        return [self.token_idx_map.get(token.lower(), self.UNK_ID) for token in sentence]

    def tags_to_idxlist(self, tags: Tags) -> List[int]:
        """
        Transform the list of tags into a list of corresponding integer tag IDs.
        Primarily used for the ground truth tags to calculate the loss.
        :param tags: a list of tags to be transformed into a list of integer tag IDs.
        :return: a list of integer tag IDs.
        """
        return [self.tag_idx_map[tag] for tag in tags]

    def idxlist_to_tags(self, idxlist: List[int]) -> Tags:
        """
        Translate a list of tag IDs into its string representation.
        :param idxlist: a list of tag IDs
        :return: a list of corresponding tags
        """
        return [self.idx_tag_map[i] for i in idxlist]

    def token_to_idxlist(self, token: str, reqlen: int) -> List[int]:
        """
        Transform the given token to a list of character IDs, padded up to the given length.
        :param reqlen: length for the returned list to be padded up to
        :return: a possibly padded list of character IDs.
        """
        return self.pad([self.char_idx_map.get(elem, self.UNK_ID) for elem in token],
                        reqlen, self.PAD_ID)

    def sentence_to_char_idxmat(self, sentence: Sentence,
                                numcols: int) -> List[List[int]]:
        """
        Transform the given sentence into a matrix of character IDs.
        Each row of the matrix represents a token of the sentence,
        while each column represents a character of the token.
        The matrix is padded columnwise upto the given column number.
        :param sentence: the sentence to be transformed.
        :param numcols: the number of columns for the matrix to be padded up to
        :return: a character ID matrix
        """
        return list(self.token_to_idxlist(token, numcols) for token in sentence)

    def __call__(self, batch: List[Sentence],
                 use_char: bool) -> Tuple[Tuple[
                     torch.Tensor, List[int]], Optional[Tuple[torch.Tensor, List[int]]]]:
        """
        Given a batch of sentences, return the (often padded) token ID tensor with its original
        lengths before padding, and optionally the (often padded) character ID tensor.
        :param batch: a batch of sentences
        :param use_char: return the character ID tensor or not
        :return:
            a tuple of a token ID tensor (batch_size, max_sentence_len) and its original unpadded
            lengths and an optional character id tensor
            (batch_size, max_sentence_len, len_longest_token).
        """
        sentence_lengths = list(len(s) for s in batch)
        max_sentence_len = max(sentence_lengths)
        padded_batch = [self.pad(sentence, max_sentence_len, '') for sentence in batch]
        token_idx_tensor = torch.tensor([self.sentence_to_idxlist(
            s) for s in padded_batch], dtype=torch.long)

        char_results: Optional[Tuple[torch.Tensor, List[int]]] = None
        if use_char:
            # b/c of the way PackedSequence works, we can't pass in a sample with 0 length.
            # this means for a padding token, we have to pretend that there was one character.
            # this shouldn't affect anything as the padding token would be ignored when calculating
            # the loss.
            token_lengths = list(
                len(token) if len(token) != 0 else 1
                for sentence in padded_batch for token in sentence)
            len_longest_token = max(token_lengths)
            char_idx_tensor = torch.tensor([self.sentence_to_char_idxmat(
                sentence, len_longest_token) for sentence in padded_batch], dtype=torch.long)
            char_results = (char_idx_tensor, token_lengths)

        return (token_idx_tensor, sentence_lengths), char_results

    def batch_to_tag_idxmat(self, batch: List[Tags]) -> torch.Tensor:
        """
        Transform the given batch of tags into a tensor of tag IDs,
        usually used for calculating the loss. The tensor is appropriately padded.
        :param batch: the batch of tags
        :return: a (often padded) tensor of tag IDs.
        """
        max_len = max(len(t) for t in batch)
        mat = [self.tags_to_idxlist(self.pad(tags, max_len, '')) for tags in batch]
        return torch.tensor(mat, dtype=torch.long)

    @staticmethod
    def pad(sequence: List[T], reqlen: int, pad: T) -> List[T]:
        """
        Pad the given sequence to the given length.
        :param sequence: the sequence to be padded
        :param reqlen: the length the given sequence to be padded to
        :param pad: padding element
        :return: the padded sequence
        """
        return sequence + [pad] * (reqlen - len(sequence))
