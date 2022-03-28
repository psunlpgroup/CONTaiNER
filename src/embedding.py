from typing import TextIO, Tuple
import numpy as np


class GloveEmbedding(dict):
    """
    Class with the pretrained 100d glove embeddings.
    Note: glove embedding is for lower case tokens.
    """
    DIM_EMDEDDING = 100

    def __init__(self, fileh: TextIO):
        """
        Initialize a GloveEmbedding instance.
        :param fileh: a file handler for the glove embeddings
        """
        super().__init__()
        self.load(fileh)

    def load(self, fileh: TextIO):
        """
        Load and parse each line of the glove embeddiings file.
        :param fileh: the glove embeddings file to be loaded
        """
        for line in fileh:
            token, embedding = GloveEmbedding.split(str(line))
            self[token.lower()] = embedding

    @staticmethod
    def split(line: str) -> Tuple[str, np.ndarray]:
        """
        Split the given line into a token and its embedding vector.
        :param line: line to be splitted into a token and its embedding vector
        :return: a tuple of a token and its embedding vector (numpy array)
        """
        token, vals = line.split(None, 1)
        return token, np.array([float(v) for v in vals.split()], dtype=np.float)

    @classmethod
    def random(cls) -> np.ndarray:
        """
        Return a random vector with the right scale.
        :return: a random numpy vector
        """
        dim = cls.DIM_EMDEDDING
        scale = np.sqrt(3.0 / dim)
        return np.random.uniform(-scale, scale, dim)

    @classmethod
    def zeros(cls) -> np.ndarray:
        """
        Return a zero vector.
        :return: a zero numpy vector
        """
        return np.zeros(cls.DIM_EMDEDDING, dtype=np.float)

    def get(self, token: str, default=None) -> np.ndarray:
        """
        Get the glove embedding if the token is found, else the given default or a random vector.
        :param token: a token to be looked up
        :param default: a default to be returned if the given token is not found
        :return: the glove embedding, default or a random vector
        """
        token = token.lower()
        ret = super(GloveEmbedding, self).get(token)
        if ret is not None:
            return ret
        elif default is not None:
            return default
        return self.random()
