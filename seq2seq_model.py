"""Seq2seq video captionnig model module"""

import numpy as np
import tensorflow as tf

class VideoCaptioningModel(object):
    """This class is used to create a video captioning neural network"""

    def __init__(self, enc_units, dec_units, rnn_layers,
                 vocab_size, embedding_dims, learning_rate, dropout_rate):
        """Keyword arguments:
        enc_units -- number of units in the encoder
        dec_units -- number of units in the decoder
        rnn_layers -- number of layers in encoder and decoder
        vocab_size -- size of the vocabulary
        embedding_dims -- size of the embedding
        learning_rate -- start learning rate used in adam optimizer
        dropout_rate -- probability to dropout in encoder
        """

        self._enc_units = enc_units
        self._dec_units = dec_units
        self._rnn_layers = rnn_layers
        self._vocab_size = vocab_size
        self._embedding_dims = embedding_dims
        self._learning_rate = learning_rate
        self._dropout_rate = dropout_rate

    def _encoder(self, input_features, videos_lengths):
        """Encoder part of the seq2seq network

        Keyword arguments:
        input_features -- video features given as input
                          Tensor [batch_size, n_frames, 2048] shaped
        videos_lengths -- length of each video
                          Tensor [batch_size] shaped"""

        with tf.name_scope("encoder"):
            def make_cell(rnn_size, keep_rate=1.0):
                """This function creates a basic cell. Used for creating several layers"""
                cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
                cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_rate)
                return cell

            # Creating the forward and backward states
            forward_cell = tf.nn.rnn_cell.MultiRNNCell(
                [make_cell(self._enc_units/2, keep_rate=1-self._dropout_rate)
                 for _ in range(self._rnn_layers)])

            backward_cell = tf.nn.rnn_cell.MultiRNNCell(
                [make_cell(self._enc_units/2, keep_rate=1-self._dropout_rate)
                 for _ in range(self._rnn_layers)])

            bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
                forward_cell, backward_cell, input_features, sequence_length=videos_lengths,
                time_major=False, dtype=tf.float32)

            # Merging forward and backward encoder outputs
            encoder_outputs = tf.concat(bi_outputs, -1)

            # Merging forward and backward encoder states
            final_state = []
            for layer_id in range(self._rnn_layers):
                state_fw = bi_state[0][layer_id] # forward state for this layer
                state_bw = bi_state[1][layer_id] # backward state for this layer

                cell_state = tf.concat([state_fw.c, state_bw.c], 1) # merging cell state
                hidden_state = tf.concat([state_fw.h, state_bw.h], 1) # merging hidden_state
                state = tf.nn.rnn_cell.LSTMStateTuple(c=cell_state, h=hidden_state)

                final_state.append(state)
            final_state = tuple(final_state)

        return encoder_outputs, final_state


    def _decoder(self, target, hidden_state, encoder_outputs, caption_lengths,
                 start_token, end_token):
        """Decoder part of the seq2seq neural network.

        This decoder is divided in two parts: train and inference
        Training decoder uses TrainingHelper (teacher's forcing mechanism) while
        inference decoder uses GreedyEmbeddingHelper (outputs are given as inputs).
        The two decoders, however, share the same weigths for the dense layer.

        The decoder uses attention mechanism.

        Keyword arguments:

        target -- tokenized and padded target caption
                  Tensor [batch_size, max_length] shaped
        hidden_state -- last hidden state of the encoder
        encoder_outputs -- all the hidden states of the encoder. Used for attention
                           Tensor [batch_size, n_frames, enc_units] shaped
        caption_lengths -- without-padding size of captions
                           Tensor [batch_size] shaped
        start_token -- code for <start> token
        end_token -- code for <end> token

        Returns:
        training_decoder_outputs -- dynamic decode object that contains logits and predictions
        inference_decoder_outputs -- dynamic decode object that contains logits and predictions
        """

        # Size of the batch
        n_data = tf.shape(target)[0]

        with tf.name_scope("decoder"):
            embeddings = tf.keras.layers.Embedding(self._vocab_size,
                                                   self._embedding_dims,
                                                   name="embeddings")
            dec_embeddings = embeddings
            decoder_inputs = embeddings(target)

            #dec_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_dims]))
            #dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, target)
            #decoder_inputs = dec_embed_input

            cell = tf.nn.rnn_cell.MultiRNNCell(
                [tf.nn.rnn_cell.BasicLSTMCell(self._dec_units)
                 for _ in range(self._rnn_layers)])

            attention_states = encoder_outputs

            attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                self._dec_units,
                attention_states)

            decoder_attention_gru_cell = tf.contrib.seq2seq.AttentionWrapper(
                cell, attention_mechanism, attention_layer_size=self._dec_units)

            decoder_initial_state = decoder_attention_gru_cell.zero_state(n_data, tf.float32)
            decoder_initial_state = decoder_initial_state.clone(cell_state=hidden_state)

            output_layer = tf.layers.Dense(self._vocab_size, kernel_initializer=
                                           tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
                                           name="decoder_dense")

            #max_effective_length = tf.reduce_max(caption_lengths)
            # Size of the longuest caption over the chosen data
            max_effective_length = tf.shape(target)[1]

            # Training decoder
            with tf.variable_scope("decoder"):
                training_helper = tf.contrib.seq2seq.TrainingHelper(decoder_inputs, caption_lengths)
                training_decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_attention_gru_cell, training_helper,
                    decoder_initial_state, output_layer)
                training_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    training_decoder, maximum_iterations=max_effective_length)

            # Inference decoder
            with tf.variable_scope("decoder", reuse=True):
                all_start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32),
                                           [n_data], name='start_tokens')

                inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    dec_embeddings, all_start_tokens, end_token)
                inference_decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_attention_gru_cell, inference_helper,
                    decoder_initial_state, output_layer)
                inference_decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    inference_decoder, maximum_iterations=max_effective_length)

        return training_decoder_outputs, inference_decoder_outputs


    def build_encoder_decoder_model(self, features, captions, video_lengths, caption_lengths,
                                    start_token, end_token):
        """This method creates encoder and decoder graphs and returns it

        Note: returns both training and inference logits. To use build_seq2seq_model
        properly, one should call the function with  a reinitializable iterator and
        switch between training and testing datasets.

        Keyword arguments:
        features -- video features
                    Tensor [batch_size, n_frames, 2048] shaped
        captions -- target captions
                    Tensor [batch_size, max_length] shaped
        video_lengths -- without-padding lengths of videos
                         Tensor [batch_size] shaped
        caption_lengths -- without-padding lengths of captions
                           Tensor [batch_size] shaped
        start_token -- code of <start> token
        end_token -- code of <end> token

        Returns:
        training_logits - inference_logits
            Tensor shaped [batch_size max_prediction_length, vocab_size]
        inference_predictions -- already decoded logits
        """

        # Reverses the videos featurees
        reversed_features = tf.reverse(features, [-1])
        # Preprocessing captions
        preprocessed_captions = preprocess_targets(captions, start_token)

        # Creating the encoder
        encoder_outputs, hidden_state = self._encoder(reversed_features, video_lengths)

        # Creating the decoder
        training_decoder_outputs, inference_decoder_outputs = self._decoder(
            preprocessed_captions, hidden_state, encoder_outputs, caption_lengths,
            start_token, end_token)

        training_logits = tf.identity(training_decoder_outputs.rnn_output,
                                      name='training_logits')
        inference_logits = tf.identity(inference_decoder_outputs.rnn_output,
                                       name='inference_logits')

        inference_predictions = tf.identity(inference_decoder_outputs.sample_id,
                                            name='inference_predictions')
        return training_logits, inference_logits, inference_predictions

    def _optimizer(self, loss):
        optimizer = tf.train.AdamOptimizer(self._learning_rate, name="optimizer")

        gradients = optimizer.compute_gradients(loss)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var)
                            for grad, var in gradients if grad is not None]

        training_op = optimizer.apply_gradients(capped_gradients, name="training_op")

        return training_op, gradients

    def training_step(self, training_logits, captions, caption_lengths):
        """Returns the elements useful to do a training step

        Keyword arguments:

        training_logits -- decoder trainig logits. Should come from build_seq2seq_model function
        captions -- targets for this training step
        caption_lengths -- without-padding lengths of the targets
        """

        loss = eval_loss(training_logits, captions, caption_lengths)
        training_op, gradients = self._optimizer(loss)

        accuracy = eval_accuracy(training_logits, captions, caption_lengths)

        return training_op, loss, accuracy, gradients

    def build_seq2seq_model(self, features, captions, video_lengths, caption_lengths,
                            start_token, end_token):
        """Builds full seq2seq model using both build_encoder_decoder_model and
        training_step functions"""

        training_logits, inference_logits, inference_predictions = self.build_encoder_decoder_model(
            features, captions, video_lengths, caption_lengths, start_token, end_token)

        training_op, loss, accuracy, _ = self.training_step(training_logits,
                                                            captions, caption_lengths)

        return training_logits, inference_logits, inference_predictions, training_op, loss, accuracy

    def model_parameters_to_dict(self):
        """Returns a dict containing all model parameters"""
        model_infos = {
            'enc_units': self._enc_units,
            'dec_units': self._dec_units,
            'rnn_layers': self._rnn_layers,
            'vocab_size': self._vocab_size,
            'embedding_dims': self._embedding_dims,
            'learning_rate': self._learning_rate,
            'dropout_rate': self._dropout_rate
        }
        return model_infos

def preprocess_targets(targets, start_token):
    """This function preprocesses targets before feeding it to the decoder"""
    # Size of batch
    n_data = tf.shape(targets)[0]

    # Cuts the last word
    prepared_captions = tf.strided_slice(targets, [0, 0], [n_data, -1])
    # Adds a <start> token
    prepared_captions = tf.concat([tf.fill([n_data, 1], start_token), prepared_captions], 1)

    return prepared_captions


def eval_loss(logits, captions, caption_lengths):
    """Returns the loss computation, given by mean of cross entropy"""
    max_length = tf.shape(captions)[1]
    paddings = [[0, 0], [0, max_length-tf.shape(logits)[1]], [0, 0]]
    padded_logits = tf.pad(logits, paddings, 'CONSTANT', constant_values=0)

    # Create the weights for sequence_loss
    masks = tf.sequence_mask(caption_lengths, max_length, dtype=tf.float32, name='masks')
    loss = tf.contrib.seq2seq.sequence_loss(logits=padded_logits, targets=captions, weights=masks,
                                            name="loss")

    return loss

def eval_accuracy(logits, captions, caption_lengths):
    """Returns the accuracy computation

    It consists of a word by word comparison between logits and targets
    We adapt the size of logits/caption to the longest element between the
    longest prediction and the longest caption of this batch
    """

    n_data = tf.shape(captions)[0]
    max_length_predicted = tf.shape(logits)[1]
    max_length_captions = tf.reduce_max(caption_lengths)

    # Determine which one is the longer
    max_length = tf.maximum(max_length_predicted, max_length_captions)

    # Crops the caption to match the size of the longest caption/prediction
    # Note: max_length <= tf.shape(captions)[1]
    # tf.shape(captions)[1] is the theoretical maximum
    cropped_captions = tf.strided_slice(captions, [0, 0], [n_data, max_length])

    # Pads the logits to match the size of the longest caption
    # If the longest prediction is longer than the longest caption, this
    # Padding operation doesn't do anything
    paddings = [[0, 0], [0, max_length-max_length_predicted], [0, 0]]
    padded_logits = tf.pad(logits, paddings, 'CONSTANT', constant_values=0)

    # Matching to the required shape of tf.nn_in_top_k
    logits = tf.reshape(logits, (-1, padded_logits.shape[2]))
    captions = tf.squeeze(tf.reshape(cropped_captions, (-1, 1)), axis=-1)

    correct = tf.nn.in_top_k(logits, captions, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

    return accuracy
