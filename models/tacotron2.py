import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell,LSTMCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, AttentionWrapper
from text.symbols import symbols
from util.infolog import log
from util.ops import shape_list
from .helpers import TacoTestHelper, TacoTrainingHelper
from .modules import encoder_cbhg, post_cbhg, prenet, reference_encoder, encoder, LocationSensitiveAttention, ZoneoutLSTMCell
from .rnn_wrappers import DecoderPrenetWrapper, ConcatOutputAndAttentionWrapper, ZoneoutWrapper
from .style_attention import MultiheadAttention


class Tacotron():
  def __init__(self, hparams):
    self._hparams = hparams


  def initialize(self, inputs, input_lengths, mel_targets=None, linear_targets=None, reference_mel=None):
    '''Initializes the model for inference.

    Sets "mel_outputs", "linear_outputs", and "alignments" fields.

    Args:
      inputs: int32 Tensor with shape [N, T_in] where N is batch size, T_in is number of
        steps in the input time series, and values are character IDs
      input_lengths: int32 Tensor with shape [N] where N is batch size and values are the lengths
        of each sequence in inputs.
      mel_targets: float32 Tensor with shape [N, T_out, M] where N is batch size, T_out is number
        of steps in the output time series, M is num_mels, and values are entries in the mel
        spectrogram. Only needed for training.
      linear_targets: float32 Tensor with shape [N, T_out, F] where N is batch_size, T_out is number
        of steps in the output time series, F is num_freq, and values are entries in the linear
        spectrogram. Only needed for training.
    '''
    with tf.variable_scope('inference') as scope:
      is_training = linear_targets is not None
      batch_size = tf.shape(inputs)[0]
      hp = self._hparams

      # Embeddings
      embedding_table = tf.get_variable(
        'text_embedding', [len(symbols), hp.embed_depth], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.5))
      embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)           # [N, T_in, 256]
      
        #Global style tokens (GST)
      gst_tokens = tf.get_variable(
        'style_tokens', [hp.num_gst, hp.style_embed_depth // hp.num_heads], dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.5))
      self.gst_tokens = gst_tokens
 
      # Encoder
      
      encoder_outputs = encoder(embedded_inputs, input_lengths, is_training, 512, 5, 256)  # [N, T_in, 256]
      
      if is_training:
        reference_mel = mel_targets

      if reference_mel is not None:
        # Reference encoder
        refnet_outputs = reference_encoder(
          reference_mel, 
          filters=hp.ref_filters, 
          kernel_size=(3,3),
          strides=(2,2),
          encoder_cell=GRUCell(hp.ref_depth),
          is_training=is_training)                                                 # [N, 128]
        self.refnet_outputs = refnet_outputs                                       

          # Style attention
        style_attention = MultiheadAttention(
          tf.expand_dims(refnet_outputs, axis=1),                                   # [N, 1, 128]
          tf.tanh(tf.tile(tf.expand_dims(gst_tokens, axis=0), [batch_size,1,1])),            # [N, hp.num_gst, 256/hp.num_heads]   
          num_heads=hp.num_heads,
          num_units=hp.style_att_dim,
          attention_type=hp.style_att_type)

        embedded_tokens = style_attention.multi_head_attention()                   # [N, 1, 256]
    
      else:
        random_weights = tf.constant(4*[[0]*(hp.gst_index-1)+[1]+[0]*(hp.num_gst-hp.gst_index)], dtype=tf.float32)
        random_weights = tf.nn.softmax(random_weights, name="random_weights")
        # gst_tokens = tf.tile(gst_tokens, [1, hp.num_heads])
        embedded_tokens = tf.matmul(random_weights, tf.nn.tanh(gst_tokens))
        embedded_tokens = hp.gst_scale*embedded_tokens
        embedded_tokens = tf.reshape(embedded_tokens, [1, 1] + [hp.num_heads * gst_tokens.get_shape().as_list()[1]])

      # Add style embedding to every text encoder state
      style_embeddings = tf.tile(embedded_tokens, [1, shape_list(encoder_outputs)[1], 1]) # [N, T_in, 128]
      encoder_outputs = tf.concat([encoder_outputs, style_embeddings], axis=-1)

      # Attention
      attention_mechanism = LocationSensitiveAttention(128, encoder_outputs,hparams=hp, is_training=is_training,
                                    mask_encoder=True, memory_sequence_length = input_lengths, smoothing=False, cumulate_weights=True)
      decoder_lstm = [ZoneoutLSTMCell(1024, is_training, zoneout_factor_cell=0.1, zoneout_factor_output=0.1, name='decoder_LSTM_{}'.format(i+1)) for i in range(2)]
            
      decoder_lstm = MultiRNNCell(decoder_lstm, state_is_tuple=True)
      decoder_init_state = decoder_lstm.zero_state(batch_size=batch_size, dtype=tf.float32) #tensorflow1에는 없음
      
      attention_cell = AttentionWrapper(decoder_lstm, attention_mechanism, initial_cell_state=decoder_init_state, alignment_history=True, output_attention=False)

      # attention_state_size = 256
      # Decoder input -> prenet -> decoder_lstm -> concat[output, attention]
      # dec_outputs = DecoderPrenetWrapper(attention_cell, is_training, hp.prenet_depths)
      dec_outputs_cell = OutputProjectionWrapper(attention_cell,(hp.num_mels) * hp.outputs_per_step)
      
      if is_training:
        helper = TacoTrainingHelper(inputs, mel_targets, hp)
      else:
        helper = TacoTestHelper(batch_size, hp)

      decoder_init_state = dec_outputs_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
      (decoder_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
        BasicDecoder(dec_outputs_cell, helper, decoder_init_state),
        maximum_iterations=hp.max_iters)                                        # [N, T_out/r, M*r]

      # Reshape outputs to be one output per entry
      decoder_mel_outputs = tf.reshape(decoder_outputs[:,:,:hp.num_mels * hp.outputs_per_step], [batch_size, -1, hp.num_mels])# [N, T_out, M]

      x = decoder_mel_outputs
      for i in range(5):
          activation = tf.nn.tanh if i != (4) else None
          x = tf.layers.conv1d(x,filters=512, kernel_size=5, padding='same', activation=activation, name='Postnet_{}'.format(i))
          x = tf.layers.batch_normalization(x, training=is_training)
          x = tf.layers.dropout(x, rate=0.5, training=is_training, name='Postnet_dropout_{}'.format(i))

      residual = tf.layers.dense(x, hp.num_mels, name='residual_projection')
      mel_outputs = decoder_mel_outputs + residual

      # Add post-processing CBHG:
      # mel_outputs: (N,T,num_mels)
      post_outputs = post_cbhg(mel_outputs, hp.num_mels, is_training)
      linear_outputs = tf.layers.dense(post_outputs, hp.num_freq)    # [N, T_out, F(1025)]             # [N, T_out, F]

      # Grab alignments from the final decoder state:
      alignments = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])

      self.inputs = inputs
      self.input_lengths = input_lengths
      self.decoder_mel_outputs = decoder_mel_outputs
      self.mel_outputs = mel_outputs
      self.encoder_outputs = encoder_outputs
      self.style_embeddings = style_embeddings
      self.linear_outputs = linear_outputs
      self.alignments = alignments
      self.mel_targets = mel_targets
      self.linear_targets = linear_targets
      self.reference_mel = reference_mel
      self.all_vars = tf.trainable_variables()
      log('Initialized Tacotron model. Dimensions: ')
      log('  text embedding:          %d' % embedded_inputs.shape[-1])
      log('  style embedding:         %d' % style_embeddings.shape[-1])
      # log('  prenet out:              %d' % prenet_outputs.shape[-1])
      log('  encoder out:             %d' % encoder_outputs.shape[-1])
      log('  attention out:           %d' % attention_cell.output_size)
      # log('  concat attn & out:       %d' % concat_cell.output_size)
      log('  decoder cell out:        %d' % dec_outputs_cell.output_size)
      log('  decoder out (%d frames):  %d' % (hp.outputs_per_step, decoder_outputs.shape[-1]))
      log('  decoder out (1 frame):   %d' % mel_outputs.shape[-1])
      log('  postnet out:             %d' % post_outputs.shape[-1])
      log('  linear out:              %d' % linear_outputs.shape[-1])


  def add_loss(self):
      '''Adds loss to the model. Sets "loss" field. initialize must have been called.'''
      with tf.variable_scope('loss') as scope:
          hp = self._hparams
          before = tf.losses.mean_squared_error(self.mel_targets, self.decoder_mel_outputs)
          after = tf.losses.mean_squared_error(self.mel_targets, self.mel_outputs)
  
          self.mel_loss = before + after


          #self.stop_token_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.stop_token_targets, logits=self.stop_token_outputs))

          l1 = tf.abs(self.linear_targets - self.linear_outputs)
          # Prioritize loss for frequencies under 3000 Hz.
          n_priority_freq = int(3000 / (hp.sample_rate * 0.5) * hp.num_freq)
          self.linear_loss = 0.5 * tf.reduce_mean(l1) + 0.5 * tf.reduce_mean(l1[:, :, 0:n_priority_freq])

          self.regularization = tf.add_n([tf.nn.l2_loss(v) for v in self.all_vars
          if not('bias' in v.name or 'Bias' in v.name or '_projection' in v.name or 'inputs_embedding' in v.name
            or 'RNN' in v.name or 'LSTM' in v.name)]) * hp.reg_weight
          self.loss = self.mel_loss + self.linear_loss + self.regularization


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
  return init_lr * warmup_steps**0.5 * tf.minimum(step * warmup_steps**-1.5, step**-0.5)
