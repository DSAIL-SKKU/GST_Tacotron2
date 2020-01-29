import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, GRUCell
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.contrib.framework import nest
from tensorflow.contrib.seq2seq.python.ops.attention_wrapper import _bahdanau_score, _BaseAttentionMechanism, BahdanauAttention, AttentionWrapperState, AttentionMechanism, _BaseMonotonicAttentionMechanism, _maybe_mask_score,_prepare_memory, _monotonic_probability_fn
from tensorflow.python.ops import array_ops, math_ops, nn_ops, variable_scope
from tensorflow.python.layers.core import Dense
import functools
_zero_state_tensors = rnn_cell_impl._zero_state_tensors

'''
Adding zoneoutLSTMcell and LocationSensitiveAttention function to existing code for Tacotron2
'''

def prenet(inputs, is_training, layer_sizes, scope=None):
	"""
	Args:
		inputs: input vector
		is_training: dropout option
		layer_sizes: iteration number
	
	Output:
		x: prenet
	"""
	x = inputs
	drop_rate = 0.5 if is_training else 0.0 # set dropout rate 0.5 (only training)
	with tf.variable_scope(scope or 'prenet'):
		for i, size in enumerate(layer_sizes): # iterate layer_sizes
			dense = tf.layers.dense(x, units=size, activation=tf.nn.relu, name='dense_%d' % (i + 1))
			x = tf.layers.dropout(dense, rate=drop_rate, training=is_training, name='dropout_%d' % (i + 1)) 
	return x


def encoder_cbhg(inputs, input_lengths, is_training, depth):
	"""
	Args:
		inputs: input tensor
		input_lengths: length of input tensor
		is_training: Batch Normalization option in Conv1D
		depth: dimensionality option of Highway net and Bidirectical GRU's output
	
	Output:
		cbhg function
	"""
	input_channels = inputs.get_shape()[2] # 3rd element of inputs' shape
	return cbhg(
		inputs,
		input_lengths,
		is_training,
		scope='encoder_cbhg',
		K=16,
		projections=[128, input_channels],
		depth=depth)


def post_cbhg(inputs, input_dim, is_training, depth):
	"""
	Args:
		inputs: input tensor
		input_dim: dimension of input tensor
		is_training: Batch Normalization option in Conv1D
		depth: dimensionality option of Highway net and Bidirectical GRU's output
	
	Output:
		cbhg function
	"""
	return cbhg(
		inputs,
		None,
		is_training,
		scope='post_cbhg',
		K=8,
		projections=[256, input_dim],
		depth=depth)


def cbhg(inputs, input_lengths, is_training, scope, K, projections, depth):
    """
    Args:
        inputs: input tensor
        input_lengths: length of input tensor
        is_training: Batch Normalization option in Conv1D
        scope: network or model name
        K: kernel size range
        projections: projection layers option
        depth: dimensionality option of Highway net and Bidirectical GRU's output
    The layers in the code are staked in the order in which they came out.
    """
    with tf.variable_scope(scope):
        with tf.variable_scope('conv_bank'):

            conv_outputs = tf.concat(
                [conv1d(inputs, k, 128, tf.nn.relu, is_training, 'conv1d_%d' % k) for k in range(1, K + 1)], #1D Convolution layers using multiple types of Convolution Kernel.
                axis=-1																						 #Iterate K with increasing filter size by 1.
            )# Convolution bank: concatenate on the last axis to stack channels from all convolutions

        # Maxpooling:
        maxpool_output = tf.layers.max_pooling1d(
            conv_outputs,
            pool_size=2,
            strides=1,
            padding='same') #1D Maxpooling layer(strides=1, width=2) 

        # Two projection layers:
        proj1_output = conv1d(maxpool_output, 3, projections[0], tf.nn.relu, is_training, 'proj_1')#1st Conv1D projections
        proj2_output = conv1d(proj1_output, 3, projections[1], None, is_training, 'proj_2')#2nd Conv1D projections

        # Residual connection:
        highway_input = proj2_output + inputs #Highway net input with residual connection

        half_depth = depth // 2
        assert half_depth * 2 == depth, 'encoder and postnet depths must be even.' #assert depth to be even

        # Handle dimensionality mismatch:
        if highway_input.shape[2] != half_depth: #check input's dimensionality and output's dimensionality are the same
            highway_input = tf.layers.dense(highway_input, half_depth) #change input's channel size to Highway net output's  size

        # 4-layer HighwayNet:
        for i in range(4):
            highway_input = highwaynet(highway_input, 'highway_%d' % (i + 1), half_depth) #make 4 Highway net layers
        rnn_input = highway_input

        # Bidirectional GRU
        outputs, states = tf.nn.bidirectional_dynamic_rnn( #make Bidirectional GRU
            GRUCell(half_depth),
            GRUCell(half_depth),
            rnn_input,
            sequence_length=input_lengths,
            dtype=tf.float32)
        return tf.concat(outputs, axis=2)  # Concat forward sequence and backward sequence

def highwaynet(inputs, scope, depth):
	with tf.variable_scope(scope):
		H = tf.layers.dense(
			inputs,
			units=depth,
			activation=tf.nn.relu,
			name='H')
		T = tf.layers.dense(
			inputs,
			units=depth,
			activation=tf.nn.sigmoid,
			name='T',
			bias_initializer=tf.constant_initializer(-1.0))
		return H * T + inputs * (1.0 - T)


def conv1d(inputs, kernel_size, channels, activation, is_training, scope):
	"""
	Args:
		inputs: input tensor
		kernel_size: length of the 1D convolution window
		channels: dimensionality of the output space
		activation: Activation function (None means linear activation)
		is_training: Batch Normalization option in Conv1D
		scope: namespace
	
	Output:
		output tensor
	"""
	with tf.variable_scope(scope):
		conv1d_output = tf.layers.conv1d( # creates a convolution kernel
			inputs,
			filters=channels,
			kernel_size=kernel_size,
			activation=activation,
			padding='same') # return output tensor
		return tf.layers.batch_normalization(conv1d_output, training=is_training)
    

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

class GmmAttention(AttentionMechanism):
    def __init__(self,
                 num_mixtures,
                 memory,
                 memory_sequence_length=None,
                 check_inner_dims_defined=True,
                 score_mask_value=None,
                 name='GmmAttention'):

        self.dtype = memory.dtype
        self.num_mixtures = num_mixtures
        self.query_layer = tf.layers.Dense(3 * num_mixtures, name='gmm_query_projection', use_bias=True, dtype=self.dtype)

        with tf.name_scope(name, 'GmmAttentionMechanismInit'):
            if score_mask_value is None:
                score_mask_value = 0.
            self._maybe_mask_score = functools.partial(
                _maybe_mask_score,
                memory_sequence_length=memory_sequence_length,
                score_mask_value=score_mask_value)
            self._value = _prepare_memory(
                memory, memory_sequence_length, check_inner_dims_defined)
            self._batch_size = (
                self._value.shape[0].value or tf.shape(self._value)[0])
            self._alignments_size = (
                    self._value.shape[1].value or tf.shape(self._value)[1])

    @property
    def values(self):
        return self._value

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def alignments_size(self):
        return self._alignments_size

    @property
    def state_size(self):
        return self.num_mixtures

    def initial_alignments(self, batch_size, dtype):
        max_time = self._alignments_size
        return _zero_state_tensors(max_time, batch_size, dtype)

    def initial_state(self, batch_size, dtype):
        state_size_ = self.state_size
        return _zero_state_tensors(state_size_, batch_size, dtype)

    def __call__(self, query, state):
        with tf.variable_scope("GmmAttention"):
            previous_kappa = state
            
            params = self.query_layer(query)   # query(dec_rnn_size=256) , params(num_mixtures(256)*3)
            alpha_hat, beta_hat, kappa_hat = tf.split(params, num_or_size_splits=3, axis=1)

            # [batch_size, num_mixtures, 1]
            alpha = tf.expand_dims(tf.exp(alpha_hat), axis=2)
            # softmax makes the alpha value more stable.
            # alpha = tf.expand_dims(tf.nn.softmax(alpha_hat, axis=1), axis=2)
            beta = tf.expand_dims(tf.exp(beta_hat), axis=2)
            kappa = tf.expand_dims(previous_kappa + tf.exp(kappa_hat), axis=2)

            # [1, 1, max_input_steps]
            mu = tf.reshape(tf.cast(tf.range(self.alignments_size), dtype=tf.float32), shape=[1, 1, self.alignments_size])  # [[[0,1,2,...]]]

            # [batch_size, max_input_steps]
            phi = tf.reduce_sum(alpha * tf.exp(-beta * (kappa - mu) ** 2.), axis=1)

        alignments = self._maybe_mask_score(phi)
        state = tf.squeeze(kappa, axis=2)

        return alignments, state