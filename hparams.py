import tensorflow as tf

# Default hyperparameters:
hparams = tf.contrib.training.HParams(
    # Comma-separated list of cleaners to run on text prior to training and eval. For non-English
    # text, you may want to use "basic_cleaners" or "transliteration_cleaners" See TRAINING_DATA.md.
    cleaners='korean_cleaners',
    # Audio:
    num_mels=80,
    num_freq=1025,
    sample_rate=21000,
    frame_length_ms=50,
    frame_shift_ms=12.5,
    preemphasis=0.97,
    min_level_db=-100,
    ref_level_db=20,

    # Encoder:
    embed_depth=256,
    embed_depth2=512,
    encoder_conv_filter=512,
    encoder_conv_kernel=5,
    encoder_stack_size=3,
    encoder_lstm_hidden_dim=256,

    #Global Style Token
    num_gst=10,
    style_embed_depth=256,
    ref_filters=[32, 32, 64, 64, 128, 128],
    ref_depth=128,
    style_att_type='mlp_attention',
    style_att_dim=128,
    gst_index=3,
    gst_scale=0.3,

    #Attention
    attention_depth=256,
    attention_filters = 32,
    attention_kernel = (31, ),
    attention_dim = 128,
    synthesis_constraint = False,
    synthesis_constraint_type = 'window',
    attention_win_size = 7,
    attention_type = 'mon_bah',
    cumulative_weights = True,
    num_heads=4,

    # Model:
    model='tacotron',
    outputs_per_step=5,
    prenet_depths=[256, 128],
    prenet_depths2=[256, 256],
    encoder_depth=256,
    postnet_depth=256,
    reg_weight = 1e-6,
    decoder_depth=256,
    RNN_type='LSTM_zoneout',
    tacotron_zoneout_rate=0.1,

    # Training:
    batch_size=32,
    adam_beta1=0.9,
    adam_beta2=0.999,
    initial_learning_rate=0.002,
    decay_learning_rate=True,
    use_cmudict=False,  # Use CMUDict during training to learn pronunciation of ARPAbet phonemes

    # Eval:
    max_iters=1000,
    griffin_lim_iters=60,
    power=1.5,  # Power to raise magnitudes to prior to Griffin-Lim
)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
