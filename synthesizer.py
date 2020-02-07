import io
import numpy as np
import tensorflow as tf
from hparams import hparams
from librosa import effects
from models import create_model
from text import text_to_sequence
from util import audio


class Synthesizer:

    def load(self, checkpoint_path, reference_mel=None, model_name='tacotron'):
        print('Constructing model: %s' % model_name)
        inputs = tf.placeholder(tf.int32, [1, None], 'inputs')
        input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
        if reference_mel is not None:
            reference_mel = tf.placeholder(tf.float32, [1, None, hparams.num_mels], 'reference_mel')

        with tf.variable_scope('model') as scope:
            self.model = create_model(model_name, hparams)
            self.model.initialize(inputs, input_lengths, reference_mel=reference_mel)
            self.wav_output = audio.inv_spectrogram_tensorflow(self.model.linear_outputs[0])
            self.alignments = self.model.alignments[0]

        print('Loading checkpoint: %s' % checkpoint_path)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(self.session, checkpoint_path)

    def synthesize(self, text, reference_mel=None, gst_index=None, gst_scale=None):
        cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
        seq = text_to_sequence(text, cleaner_names)
        feed_dict = {
            self.model.inputs: [np.asarray(seq, dtype=np.int32)],
            self.model.input_lengths: np.asarray([len(seq)], dtype=np.int32)
        }
        if reference_mel is not None:
            reference_mel = np.expand_dims(reference_mel, 0)
            feed_dict.update({self.model.reference_mel: np.asarray(reference_mel, dtype=np.float32)})
        
        wav= self.session.run(self.wav_output, feed_dict=feed_dict)
        wav = audio.inv_preemphasis(wav)
        wav = wav[:audio.find_endpoint(wav)]
        out = io.BytesIO()
        audio.save_wav(wav, out)
        return out.getvalue()
