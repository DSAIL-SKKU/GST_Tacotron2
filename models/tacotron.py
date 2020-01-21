import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper, LSTMCell
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, AttentionWrapper
from text.symbols import symbols
from util.infolog import log
from .helpers import TacoTestHelper, TacoTrainingHelper
from .modules import encoder_cbhg, post_cbhg, prenet
from .rnn_wrappers import DecoderPrenetWrapper, ConcatOutputAndAttentionWrapper


class Tacotron():
    def __init__(self, hparams):
        self._hparams = hparams

    def initialize(self, inputs, input_lengths, mel_targets=None, linear_targets=None):

        with tf.variable_scope('embedding') as scope:
            is_training = linear_targets is not None
            batch_size = tf.shape(inputs)[0]
            hp = self._hparams

            # Embeddings
            embedding_table = tf.get_variable(
                'embedding', [len(symbols), hp.embed_depth], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.5))
            embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)  # [N, T_in, embed_depth=512]

        with tf.variable_scope('encoder') as scope:
            x = embedded_inputs
            for i in range(hp.encoder_stack_size):
                x = tf.layers.conv1d(x,
                                     filters=hp.encoder_conv_filter,
                                     kernel_size=hp.encoder_conv_kernel,
                                     padding='same',
                                     activation=tf.nn.relu)
                x = tf.layers.batch_normalization(x, training=is_training)

            lstm_fw = LSTMCell(hp.encoder_lstm_hidden_dim)
            lstm_bw = LSTMCell(hp.encoder_lstm_hidden_dim)

            encoder_conv_output = x
            outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw,
                                                              lstm_bw,
                                                              encoder_conv_output,
                                                              sequence_length=input_lengths) # [N, T_in, 512]
            encoder_output = tf.concat(outputs, axis=2)

        # with tf.variable_scope('decoder') as scope:

			
        self.inputs = inputs
        self.input_lengths = input_lengths
        # self.mel_outputs = mel_outputs
        # self.linear_outputs = linear_outputs
        # self.alignments = alignments
        self.mel_targets = mel_targets
        self.linear_targets = linear_targets
        log('Initialized Tacotron model. Dimensions: ')
        log('  embedding:               %d' % embedded_inputs.shape[-1])
        log('  encoder out:             %d' % encoder_output.shape[-1])
        # log('  attention out:           %d' % attention_cell.output_size)
        # log('  concat attn & out:       %d' % concat_cell.output_size)
        # log('  decoder cell out:        %d' % decoder_cell.output_size)
        # log('  decoder out (%d frames):  %d' % (hp.outputs_per_step, decoder_outputs.shape[-1]))
        # log('  decoder out (1 frame):   %d' % mel_outputs.shape[-1])
        # log('  postnet out:             %d' % post_outputs.shape[-1])
        # log('  linear out:              %d' % linear_outputs.shape[-1])

    def add_loss(self):
        '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
        with tf.variable_scope('loss') as scope:
            hp = self._hparams
            self.mel_loss = tf.reduce_mean(tf.abs(self.mel_targets - self.mel_outputs))
            l1 = tf.abs(self.linear_targets - self.linear_outputs)
            # Prioritize loss for frequencies under 3000 Hz.
            n_priority_freq = int(3000 / (hp.sample_rate * 0.5) * hp.num_freq)
            self.linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:, :, 0:n_priority_freq])
            self.loss = self.mel_loss + self.linear_loss

    def add_optimizer(self, global_step):
        '''Adds optimizer. Sets "gradients" and "optimize" fields. add_loss must have been called.

        Args:
          global_step: int32 scalar Tensor representing current global step in training
        '''
        with tf.variable_scope('optimizer') as scope:
            hp = self._hparams
            if hp.decay_learning_rate:
                self.learning_rate = _learning_rate_decay(hp.initial_learning_rate, global_step)
            else:
                self.learning_rate = tf.convert_to_tensor(hp.initial_learning_rate)
            optimizer = tf.train.AdamOptimizer(self.learning_rate, hp.adam_beta1, hp.adam_beta2)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            self.gradients = gradients
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, 1.0)

            # Add dependency on UPDATE_OPS; otherwise batchnorm won't work correctly. See:
            # https://github.com/tensorflow/tensorflow/issues/1122
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.optimize = optimizer.apply_gradients(zip(clipped_gradients, variables),
                                                          global_step=global_step)


def _learning_rate_decay(init_lr, global_step):
    # Noam scheme from tensor2tensor:
    warmup_steps = 4000.0
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)
