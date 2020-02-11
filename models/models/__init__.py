from .tacotron2 import Tacotron
# from .tacotron import Tacotron


def create_model(name, hparams):
  if hparams.model == 'tacotron2':
    return Tacotron(hparams)
  # elif hparams.model == 'tacotron2':
  #   return Tacotron2(hparams)
  else:
    raise Exception('Unknown model: ' + name)
