import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
from .modules import prenet

'''
Adding zoneoutLSTMcell and LocationSensitiveAttention function to existing code for Tacotron2
'''

class ZoneoutLSTMCell(RNNCell):
    '''Wrapper for tf LSTM to create Zoneout LSTM Cell
    inspired by:
    https://github.com/teganmaharaj/zoneout/blob/master/zoneout_tensorflow.py
    Published by one of 'https://arxiv.org/pdf/1606.01305.pdf' paper writers.
    '''
    def __init__(self, num_units, is_training, zoneout_factor_cell=0., zoneout_factor_output=0., state_is_tuple=True, name=None):
        '''Initializer with possibility to set different zoneout values for cell/hidden states.
        '''
        zm = min(zoneout_factor_output, zoneout_factor_cell)
        zs = max(zoneout_factor_output, zoneout_factor_cell)

        if zm < 0. or zs > 1.:
            raise ValueError('One/both provided Zoneout factors are not in [0, 1]')

        self._cell = tf.nn.rnn_cell.LSTMCell(num_units, state_is_tuple=state_is_tuple, name=name)
        self._zoneout_cell = zoneout_factor_cell
        self._zoneout_outputs = zoneout_factor_output
        self.is_training = is_training
        self.state_is_tuple = state_is_tuple

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        '''Runs vanilla LSTM Cell and applies zoneout.
        '''
        #Apply vanilla LSTM
        output, new_state = self._cell(inputs, state, scope)

        if self.state_is_tuple:
            (prev_c, prev_h) = state
            (new_c, new_h) = new_state
        else:
            num_proj = self._cell._num_units if self._cell._num_proj is None else self._cell._num_proj
            prev_c = tf.slice(state, [0, 0], [-1, self._cell._num_units])
            prev_h = tf.slice(state, [0, self._cell._num_units], [-1, num_proj])
            new_c = tf.slice(new_state, [0, 0], [-1, self._cell._num_units])
            new_h = tf.slice(new_state, [0, self._cell._num_units], [-1, num_proj])

        #Apply zoneout
        if self.is_training:
            #nn.dropout takes keep_prob (probability to keep activations) not drop_prob (probability to mask activations)!
            c = (1 - self._zoneout_cell) * tf.nn.dropout(new_c - prev_c, (1 - self._zoneout_cell)) + prev_c   # tf.nn.dropout outputs the input element scaled up by 1 / keep_prob
            h = (1 - self._zoneout_outputs) * tf.nn.dropout(new_h - prev_h, (1 - self._zoneout_outputs)) + prev_h

        else:
            c = (1 - self._zoneout_cell) * new_c + self._zoneout_cell * prev_c
            h = (1 - self._zoneout_outputs) * new_h + self._zoneout_outputs * prev_h

        new_state = tf.nn.rnn_cell.LSTMStateTuple(c, h) if self.state_is_tuple else tf.concat(1, [c, h])

        return output, new_state

class DecoderPrenetWrapper(RNNCell):
    '''Runs RNN inputs through a prenet before sending them to the cell.'''

    def __init__(self, cell, is_training, layer_sizes):
        super(DecoderPrenetWrapper, self).__init__()
        self._cell = cell
        self._is_training = is_training
        self._layer_sizes = layer_sizes

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def call(self, inputs, state):
        prenet_out = prenet(inputs, self._is_training, self._layer_sizes, scope='decoder_prenet')
        return self._cell(prenet_out, state)

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)


class ConcatOutputAndAttentionWrapper(RNNCell):
    '''Concatenates RNN cell output with the attention context vector.

  This is expected to wrap a cell wrapped with an AttentionWrapper constructed with
  attention_layer_size=None and output_attention=False. Such a cell's state will include an
  "attention" field that is the context vector.
  '''

    def __init__(self, cell):
        super(ConcatOutputAndAttentionWrapper, self).__init__()
        self._cell = cell

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size + self._cell.state_size.attention

    def call(self, inputs, state):
        output, res_state = self._cell(inputs, state)
        return tf.concat([output, res_state.attention], axis=-1), res_state

    def zero_state(self, batch_size, dtype):
        return self._cell.zero_state(batch_size, dtype)
    
class LocationSensitiveAttention(BahdanauAttention):
    """Impelements Bahdanau-style (cumulative) scoring function.
    Usually referred to as "hybrid" attention (content-based + location-based)
    Extends the additive attention described in:
    "D. Bahdanau, K. Cho, and Y. Bengio, “Neural machine transla-
  tion by jointly learning to align and translate,” in Proceedings
  of ICLR, 2015."
    to use previous alignments as additional location features.
    This attention is described in:
    J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
  gio, “Attention-based models for speech recognition,” in Ad-
  vances in Neural Information Processing Systems, 2015, pp.
  577–585.
    """

    def __init__(self,
                 num_units,
                 memory,
                 hparams,
                 is_training,
                 mask_encoder=True,
                 memory_sequence_length=None,
                 smoothing=False,
                 cumulate_weights=True,
                 name='LocationSensitiveAttention'):
        """Construct the Attention mechanism.
        Args:
            num_units: The depth of the query mechanism.
            memory: The memory to query; usually the output of an RNN encoder.  This
                tensor should be shaped `[batch_size, max_time, ...]`.
            mask_encoder (optional): Boolean, whether to mask encoder paddings.
            memory_sequence_length (optional): Sequence lengths for the batch entries
                in memory.  If provided, the memory tensor rows are masked with zeros
                for values past the respective sequence lengths. Only relevant if mask_encoder = True.
            smoothing (optional): Boolean. Determines which normalization function to use.
                Default normalization function (probablity_fn) is softmax. If smoothing is
                enabled, we replace softmax with:
                        a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
                Introduced in:
                    J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
                  gio, “Attention-based models for speech recognition,” in Ad-
                  vances in Neural Information Processing Systems, 2015, pp.
                  577–585.
                This is mainly used if the model wants to attend to multiple input parts
                at the same decoding step. We probably won't be using it since multiple sound
                frames may depend on the same character/phone, probably not the way around.
                Note:
                    We still keep it implemented in case we want to test it. They used it in the
                    paper in the context of speech recognition, where one phoneme may depend on
                    multiple subsequent sound frames.
            name: Name to use when creating ops.
        """
        #Create normalization function
        #Setting it to None defaults in using softmax
        normalization_function = _smoothing_normalization if (smoothing == True) else None
        memory_length = memory_sequence_length if (mask_encoder==True) else None
        super(LocationSensitiveAttention, self).__init__(
                num_units=num_units,
                memory=memory,
                memory_sequence_length=memory_length,
                probability_fn=normalization_function,
                name=name)

        self.location_convolution = tf.layers.Conv1D(filters=hparams.attention_filters,
            kernel_size=hparams.attention_kernel, padding='same', use_bias=True,
            bias_initializer=tf.zeros_initializer(), name='location_features_convolution')
        self.location_layer = tf.layers.Dense(units=num_units, use_bias=False,dtype=tf.float32, name='location_features_projection')
        self._cumulate = cumulate_weights
        self.synthesis_constraint = hparams.synthesis_constraint and not is_training
        self.attention_win_size = tf.convert_to_tensor(hparams.attention_win_size, dtype=tf.int32)
        self.constraint_type = hparams.synthesis_constraint_type

    def __call__(self, query, state):
        """Score the query based on the keys and values.
        Args:
            query: Tensor of dtype matching `self.values` and shape
                `[batch_size, query_depth]`.
            state (previous alignments): Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]`
                (`alignments_size` is memory's `max_time`).
        Returns:
            alignments: Tensor of dtype matching `self.values` and shape
                `[batch_size, alignments_size]` (`alignments_size` is memory's
                `max_time`).
        """
        previous_alignments = state
        with variable_scope.variable_scope(None, "Location_Sensitive_Attention", [query]):

            # processed_query shape [batch_size, query_depth] -> [batch_size, attention_dim]
            processed_query = self.query_layer(query) if self.query_layer else query
            # -> [batch_size, 1, attention_dim]
            processed_query = tf.expand_dims(processed_query, 1)

            # processed_location_features shape [batch_size, max_time, attention dimension]
            # [batch_size, max_time] -> [batch_size, max_time, 1]
            expanded_alignments = tf.expand_dims(previous_alignments, axis=2)
            # location features [batch_size, max_time, filters]
            f = self.location_convolution(expanded_alignments)
            # Projected location features [batch_size, max_time, attention_dim]
            processed_location_features = self.location_layer(f)

            # energy shape [batch_size, max_time]
            energy = _location_sensitive_score(processed_query, processed_location_features, self.keys)

        if self.synthesis_constraint:
            prev_max_attentions = tf.argmax(previous_alignments, -1, output_type=tf.int32)
            Tx = tf.shape(energy)[-1]
            # prev_max_attentions = tf.squeeze(prev_max_attentions, [-1])
            if self.constraint_type == 'monotonic':
                key_masks = tf.sequence_mask(prev_max_attentions, Tx)
                reverse_masks = tf.sequence_mask(Tx - self.attention_win_size - prev_max_attentions, Tx)[:, ::-1]
            else:
                assert self.constraint_type == 'window'
                key_masks = tf.sequence_mask(prev_max_attentions - (self.attention_win_size // 2 + (self.attention_win_size % 2 != 0)), Tx)
                reverse_masks = tf.sequence_mask(Tx - (self.attention_win_size // 2) - prev_max_attentions, Tx)[:, ::-1]
            
            masks = tf.logical_or(key_masks, reverse_masks)
            paddings = tf.ones_like(energy) * (-2 ** 32 + 1)  # (N, Ty/r, Tx)
            energy = tf.where(tf.equal(masks, False), energy, paddings)

        # alignments shape = energy shape = [batch_size, max_time]
        alignments = self._probability_fn(energy, previous_alignments)

        # Cumulate alignments
        if self._cumulate:
            next_state = alignments + previous_alignments
        else:
            next_state = alignments

        return alignments, next_state


def _location_sensitive_score(W_query, W_fil, W_keys):
    """Impelements Bahdanau-style (cumulative) scoring function.
    This attention is described in:
        J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
      gio, “Attention-based models for speech recognition,” in Ad-
      vances in Neural Information Processing Systems, 2015, pp.
      577–585.
    #############################################################################
              hybrid attention (content-based + location-based)
                               f = F * α_{i-1}
       energy = dot(v_a, tanh(W_keys(h_enc) + W_query(h_dec) + W_fil(f) + b_a))
    #############################################################################
    Args:
        W_query: Tensor, shape '[batch_size, 1, attention_dim]' to compare to location features.
        W_location: processed previous alignments into location features, shape '[batch_size, max_time, attention_dim]'
        W_keys: Tensor, shape '[batch_size, max_time, attention_dim]', typically the encoder outputs.
    Returns:
        A '[batch_size, max_time]' attention score (energy)
    """
    # Get the number of hidden units from the trailing dimension of keys
    dtype = W_query.dtype
    num_units = W_keys.shape[-1].value or array_ops.shape(W_keys)[-1]

    v_a = tf.get_variable(
        'attention_variable_projection', shape=[num_units], dtype=dtype,
        initializer=tf.contrib.layers.xavier_initializer())
    b_a = tf.get_variable(
        'attention_bias', shape=[num_units], dtype=dtype,
        initializer=tf.zeros_initializer())

    return tf.reduce_sum(v_a * tf.tanh(W_keys + W_query + W_fil + b_a), [2])


def _smoothing_normalization(e):
    """Applies a smoothing normalization function instead of softmax
    Introduced in:
        J. K. Chorowski, D. Bahdanau, D. Serdyuk, K. Cho, and Y. Ben-
      gio, “Attention-based models for speech recognition,” in Ad-
      vances in Neural Information Processing Systems, 2015, pp.
      577–585.
    ############################################################################
                        Smoothing normalization function
                a_{i, j} = sigmoid(e_{i, j}) / sum_j(sigmoid(e_{i, j}))
    ############################################################################
    Args:
        e: matrix [batch_size, max_time(memory_time)]: expected to be energy (score)
            values of an attention mechanism
    Returns:
        matrix [batch_size, max_time]: [0, 1] normalized alignments with possible
            attendance to multiple memory time steps.
    """
    return tf.nn.sigmoid(e) / tf.reduce_sum(tf.nn.sigmoid(e), axis=-1, keepdims=True)
