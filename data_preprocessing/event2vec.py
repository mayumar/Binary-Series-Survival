from typing import Sequence, Optional, Tuple, List

import numpy as np
from gensim.models import Word2Vec

def train_event2vec_from_binary_tokens(
    binary_tokens: Sequence[Sequence[str]],
    embedding_dim: int = 32,
    window: int = 90, 
    min_count: int = 1,
    sg: int = 1,
    workers: int = 4,
    epochs: int = 20
) -> Word2Vec:
    """
    Train a Word2Vec model (Event2Vec) on binary token sequences.

    :param binary_tokens: List of token sequences (sentences). Each inner list is one window/sentence.
    :type binary_tokens: Sequence[Sequence[str]]
    :param embedding_dim: Embedding dimensionality.
    :type embedding_dim: int
    :param window: Context window for Word2Vec.
    :type window: int
    :param min_count: Minimum token frequency.
    :type min_count: int
    :param sg: Training algorithm (1=skip-gram, 0=CBOW).
    :type sg: int
    :param workers: Number of worker threads.
    :type workers: int
    :param epochs: Training epochs.
    :type epochs: int
    :return: Trained Word2Vec model.
    :rtype: Word2Vec
    """
    print("word2vec start")
        
    model = Word2Vec(
        sentences=binary_tokens,
        vector_size=embedding_dim,
        window=window,
        min_count=min_count,
        sg=sg,         
        workers=workers
    )

    model.train(
        binary_tokens,
        total_examples=len(binary_tokens),
        epochs=epochs
    )

    print("word2vec end")

    return model


def tokens_to_embeddings(tokens: Sequence[str], model: Word2Vec) -> np.ndarray:
    """
    Convert a token window into an embedding matrix using a trained Word2Vec model.

    :param tokens: Token sequence for a single window.
    :type tokens: Sequence[str]
    :param model: Trained Word2Vec model.
    :type model: Word2Vec
    :return: Array of shape (len(tokens), model.vector_size).
    :rtype: np.ndarray
    """
    emb_dim = model.vector_size
    X = np.zeros((len(tokens), emb_dim))
    for i, t in enumerate(tokens):
        if t in model.wv:
            X[i] = model.wv[t]
    return X


def tokens_list_to_embeddings(
    tokens_list: Sequence[Sequence[str]],
    window_size: int,
    model: Optional[Word2Vec] = None
) -> Tuple[List[np.ndarray], Word2Vec]:
    """
    Convert a list of token windows into a list of embedding matrices.

    If ``model`` is None, a new Word2Vec model is trained from ``tokens_list``.

    :param tokens_list: List of token windows (each window is a list of tokens).
    :type tokens_list: Sequence[Sequence[str]]
    :param window_size: Window size (used when training a new model).
    :type window_size: int
    :param model: Optional pre-trained Word2Vec model.
    :type model: Optional[Word2Vec]
    :return: Tuple (embeddings_per_window, model).
    :rtype: Tuple[List[numpy.ndarray], Word2Vec]
    """
    if model is None:
        model = train_event2vec_from_binary_tokens(tokens_list, window=window_size)

    embeddings = []
    for tokens in tokens_list:
        x = tokens_to_embeddings(tokens, model)
        embeddings.append(x)
    
    return embeddings, model