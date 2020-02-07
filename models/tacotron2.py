import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, OutputProjectionWrapper, ResidualWrapper
from tensorflow.contrib.seq2seq import BasicDecoder, BahdanauAttention, AttentionWrapper, BahdanauMonotonicAttention, LuongAttention
from text.symbols import symbols
from util.infolog import log
from util.ops import shape_list
from .helpers import TacoTestHelper, TacoTrainingHelper
from .modules import encoder, reference_encoder, post_cbhg, prenet, LocationSensitiveAttention, ZoneoutLSTMCell, GmmAttention, BahdanauStepwiseMonotonicAttention
from .rnn_wrappers import DecoderPrenetWrapper
from .style_attention import MultiheadAttention

class Tacotron2():
    def __init__(self, hparams):
        self._hparams = hparams

    def initialize(self, inputs, input_lengths, mel_targets=None, linear_targets=None, referece_mel=None):
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
                'embedding', [len(symbols), hp.embed_depth2], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.5))
            
            embedded_inputs = tf.nn.embedding_lookup(embedding_table, inputs)  # [N, T_in, embed_depth=256]
            
            #Global Style Token Embeddings
            gst_tokens = tf.get_variable('style_tokens', [hp.num_gst, hp.style_embed_depth//hp.num_heads], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=0.5))    #[]
            self.gst_tokens = gst_tokens
   
            #Enocer
            encoder_outputs = encoder(embedded_inputs, input_lengths, is_training, 512, 5, 256)

            #Reference Encoder
            if is_training:
                reference_mel = mel_targets
            
            if reference_mel is not None:
                ref_outputs = reference_encoder(reference_mel, hp.ref_filters, (3,3), (2,2), GRUCell(hp.ref_depth), is_training)
                self.ref_outputs = ref_outputs

                #Style Attention
                style_attention = MultiheadAttention(tf.expand_dims(ref_outputs, axis=1), 
                    tf.tanh(tf.tile(tf.expand_dims(gst_tokens, axis=0), [batch_size,1,1])), 
                    hp.num_heads, hp.style_att_type, hp.style_att_dim)
                
                embedded_tokens = style_attention.multi_head_attention()
            else:
                random_weights = tf.constant(4*[[0]*(hp.gst_index-1)+[1]+[0]*(hp.num_gst-hp.gst_index)], dtype=tf.float32)
                # random_weights = tf.nn.softmax(random_weights, name="random_weights")
                # gst_tokens = tf.tile(gst_tokens, [1, hp.num_heads])
                embedded_tokens = tf.matmul(random_weights, tf.nn.tanh(gst_tokens))
                embedded_tokens = hp.gst_scale*embedded_tokens
                style_embeddings = tf.reshape(embedded_tokens, [1, 1] + [hp.num_heads * gst_tokens.get_shape().as_list()[1]])

            # Add style embedding to every text encoder state
            style_embeddings = tf.tile(embedded_tokens, [1, shape_list(encoder_outputs)[1], 1]) # [N, T_in, 128]
            encoder_outputs = tf.concat([encoder_outputs, style_embeddings], axis=-1)
  
            
            if hp.attention_type == 'loc_sen': # Location Sensitivity Attention
                attention_mechanism = LocationSensitiveAttention(128, encoder_outputs,hparams=hp, is_training=is_training,
                                    mask_encoder=True, memory_sequence_length = input_lengths, smoothing=False, cumulate_weights=True)
            elif hp.attention_type == 'gmm': # GMM Attention
                attention_mechanism = GmmAttention(128, memory=encoder_outputs, memory_sequence_length = input_lengths) 
            elif hp.attention_type == 'step_bah': # Stepwise 
                attention_mechanism = BahdanauStepwiseMonotonicAttention(128, encoder_outputs, memory_sequence_length = input_lengths, mode="parallel")
            elif hp.attention_type == 'mon_bah': # Monotonic Attention
                attention_mechanism = BahdanauMonotonicAttention(128, encoder_outputs, memory_sequence_length = input_lengths, normalize=True)
            elif hp.attention_type == 'loung': # Loung Attention
                attention_mechanism = LuongAttention(128, encoder_outputs, memory_sequence_length = input_lengths, scale=True) 

            #mask_encoder: whether to mask encoder padding while computing location sensitive attention. Set to True for better prosody but slower convergence.
            #cumulate_weights: Whether to cumulate (sum) all previous attention weights or simply feed previous weights (Recommended: True)
            
            decoder_lstm = [ZoneoutLSTMCell(1024, is_training, zoneout_factor_cell=0.1, zoneout_factor_output=0.1, name='decoder_LSTM_{}'.format(i+1)) for i in range(2)]
            
            decoder_lstm = tf.contrib.rnn.MultiRNNCell(decoder_lstm, state_is_tuple=True)
            decoder_init_state = decoder_lstm.zero_state(batch_size=batch_size, dtype=tf.float32)
            
            attention_cell = AttentionWrapper(decoder_lstm, attention_mechanism, initial_cell_state=decoder_init_state, alignment_history=True, output_attention=False)

            # attention_state_size = 256
            # Decoder input -> prenet -> decoder_lstm -> concat[output, attention]
            dec_outputs = DecoderPrenetWrapper(attention_cell, is_training, hp.prenet_depths2)
            dec_outputs_cell = OutputProjectionWrapper(dec_outputs, (hp.num_mels) * hp.outputs_per_step)

            if is_training:
                helper = TacoTrainingHelper(inputs, mel_targets, hp.num_mels, hp.outputs_per_step)
            else:
                helper = TacoTestHelper(batch_size, hp.num_mels, hp.outputs_per_step)
                
            decoder_init_state = dec_outputs_cell.zero_state(batch_size=batch_size, dtype=tf.float32)
            (decoder_outputs, _), final_decoder_state, _ = tf.contrib.seq2seq.dynamic_decode(
                BasicDecoder(dec_outputs_cell, helper, decoder_init_state),
                maximum_iterations=hp.max_iters)  # [N, T_out/r, M*r]

            # Reshape outputs to be one output per entry
            decoder_mel_outputs = tf.reshape(decoder_outputs[:,:,:hp.num_mels * hp.outputs_per_step], [batch_size, -1, hp.num_mels])  # [N, T_out, M]
            #stop_token_outputs = tf.reshape(decoder_outputs[:,:,hp.num_mels * hp.outputs_per_step:], [batch_size, -1]) # [N,iters]
            
            # Postnet
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
            post_outputs = post_cbhg(mel_outputs, hp.num_mels, is_training, hp.postnet_depth)
            linear_outputs = tf.layers.dense(post_outputs, hp.num_freq)    # [N, T_out, F(1025)]
 
            # Grab alignments from the final decoder state:
            alignments = tf.transpose(final_decoder_state.alignment_history.stack(), [1, 2, 0])  # batch_size, text length(encoder), target length(decoder)
 
			
            self.inputs = inputs
            self.input_lengths = input_lengths
            self.decoder_mel_outputs = decoder_mel_outputs
            self.encoder_outputs = encoder_outputs
            self.style_embeddings = style_embeddings
            self.mel_outputs = mel_outputs
            self.linear_outputs = linear_outputs
            self.alignments = alignments
            self.mel_targets = mel_targets
            self.linear_targets = linear_targets
            self.referece_mel = referece_mel
            #self.stop_token_targets = stop_token_targets
            #self.stop_token_outputs = stop_token_outputs
            self.all_vars = tf.trainable_variables()
            log('Initialized Tacotron model. Dimensions: ')
            log('  text embedding:               %d' % embedded_inputs.shape[-1])
            log('  style embedding:              %d' % style_embeddings.shape[-1])
            log('  encoder out:             %d' % encoder_outputs.shape[-1])
            log('  attention out:           %d' % attention_cell.output_size)
            #log('  concat attn & out:       %d' % concat_cell.output_size)
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
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)
