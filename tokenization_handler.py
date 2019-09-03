"""This module contains TokenizationHandler class used to handle tokenization"""

import tensorflow as tf
import numpy as np

def build_tokenizer(captions):
    """This method builds a keras tokenizer from a list of captions

    Uses the vocabulary contains in the captions
    Adds <start> and <pad> tokens at the vocabulary
    """

    # Create a Tokenizer fitted on captions
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(captions)

    vocab_size = len(lang_tokenizer.word_index)

    # Incorporating <start> and <pad> tokens
    vocab_size = vocab_size + 1
    lang_tokenizer.word_index['<pad>'] = vocab_size
    lang_tokenizer.index_word[vocab_size] = '<pad>'

    vocab_size = vocab_size + 1
    lang_tokenizer.word_index['<start>'] = vocab_size
    lang_tokenizer.index_word[vocab_size] = '<start>'

    vocab_size += 1

    return lang_tokenizer, vocab_size

class TokenizationHandler(object):
    """Handle caption tokenization over a list of captions defined vocabulary

    Creates a keras tokenizer
    This keras tokenizer is crated with given list of captions
    Be carefull: <pad> and <start> tokens are automatically added to the vocabulary
    """

    def __init__(self, captions):
        """Inits the tokenizer from its list of captions"""
        # Creates the tokenizer from the lsit of captions
        self._tokenizer, self._vocab_size = build_tokenizer(captions)

        # Computes the max length over all the captions
        # This max length value is then used for caption padding
        self._max_length = max([len(caption.split(' ')) for caption in captions])

    def tokenize_captions(self, captions):
        """Tokenizes the given list of captions

        Padds it with <pad> tokens according to the biggest caption length used
        to create the tokenizer (max_length)
        """

        tensor = self._tokenizer.texts_to_sequences(captions)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(
            tensor, maxlen=self._max_length, padding='post',
            value=self._tokenizer.word_index['<pad>'])

        return np.asarray(tensor)

    def from_tokens_to_sentence(self, tokens, stop_end=False):
        """Converts a sequence of tokens to a sentence

        If stop_end argument is set to True, the translation stops after the first
        encountered <end> token
        """

        end_token = self._tokenizer.word_index['<end>']
        string = ''
        for token in tokens:
            if token == end_token and stop_end:
                return string
            string = string + self._tokenizer.index_word[token] + ' '
        return string

    def get_token(self, word):
        """Converts a word into the associated token"""
        try:
            return self._tokenizer.word_index[word]
        except KeyError:
            print(word + ' is not in the vocabulary')
            return None

    def get_vocab_size(self):
        """Returns the size of the vocabulary"""
        return self._vocab_size

    def get_padding_token(self):
        """Returns the <pad> token created by the tokenizer"""
        return self._tokenizer.word_index['<pad>']

    def get_start_token(self):
        """Returns the <start> token created by the tokenizer"""
        return self._tokenizer.word_index['<start>']
