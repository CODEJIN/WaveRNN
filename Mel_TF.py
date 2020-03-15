import tensorflow as tf
import numpy as np
import json, librosa

def melspectrogram(signals, num_freq, num_mels, hop_length, win_length, sample_rate, max_abs_value = None):
    mel_weights = _linear_to_mel(num_freq, num_mels, sample_rate)
    
    signals_cast = tf.cast(signals, dtype=tf.float32)    #stft does not support float16
    stft = _stft(signals_cast, num_freq, hop_length, win_length, sample_rate)
    magnitude = tf.abs(stft)
    mel = _amp_to_db(tf.matmul(tf.transpose(magnitude), mel_weights))    
    mel_norm = _normalize(mel) if max_abs_value is None else _symmetric_normalize(mel, max_abs_value= max_abs_value)

    return tf.cast(mel_norm, dtype= signals.dtype)

def _stft(signals, num_freq, hop_length, win_length, sample_rate):
    n_fft = (num_freq - 1) * 2
    return tf.transpose(tf.signal.stft(
        signals, win_length, hop_length, fft_length= n_fft,
        window_fn=tf.signal.hann_window, pad_end=True
        ))

def _linear_to_mel(num_freq, num_mel, sample_rate):
    mel_f = librosa.mel_frequencies(num_mel + 2)
    enorm = 2.0 / (mel_f[2:num_mel+2] - mel_f[:num_mel])
    return tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins= num_mel, num_spectrogram_bins=num_freq, sample_rate=sample_rate,
        lower_edge_hertz=0.0, upper_edge_hertz= sample_rate / 2
        ) * enorm

def _amp_to_db(x):
    return 20 * tf_log10(tf.maximum(x, 1e-5))

def _normalize(S, min_level_db = -100):
    return tf.clip_by_value((S - min_level_db) / -min_level_db, 0, 1)

def _symmetric_normalize(S, min_level_db = -100, max_abs_value = 4):
    return tf.clip_by_value((2 * max_abs_value) * ((S - min_level_db) / (-min_level_db)) - max_abs_value, -max_abs_value, max_abs_value)

def tf_log10(x):
    return tf.math.log(x) / tf.math.log(10.0)


if __name__ == "__main__":    
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]= '1'

    sig1 = librosa.load('D:/Python_Programming/WaveGlow/Wav_for_Inference/LJ.LJ050-0278.wav', 16000)[0]    
    x = librosa.load('D:/Python_Programming/WaveGlow/Wav_for_Inference/FV.CLB.arctic_a0003.wav', 16000)[0]
    sig2 = np.zeros_like(sig1)
    sig2[:x.shape[0]]  = x
    sig = np.stack([sig1, sig2], axis=0)
    mel = melspectrogram(sig, 513, 80, 12.5, 50, 16000, 4)
    
    import matplotlib.pyplot as plt
    plt.subplot(211)
    plt.imshow(tf.transpose(mel[0]), aspect='auto', origin='lower')
    plt.colorbar()
    plt.subplot(212)
    plt.imshow(tf.transpose(mel[1]), aspect='auto', origin='lower')
    plt.colorbar()
    plt.show()