import os
import numpy as np
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt
# path_List = []
# for root, _, files in os.walk('C:/Pattern/LJSpeech/wavs'):
#     path_List.extend([
#         os.path.join(root, file).replace('\\', '/')
#         for file in files
#         if os.path.splitext(file)[1].upper() == '.WAV'
#         ])

# os.makedirs('C:/Pattern/LJSpeech_16000/wavs', exist_ok= True)

# for path in path_List:
#     sig = librosa.core.load(
#         path,
#         sr = 16000
#         )[0]
#     wavfile.write(path.replace('LJSpeech', 'LJSpeech_16000'), 16000, (sig * 32768).astype(np.int16))

# import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"]= '1'
# def time_to_batch(value, dilation, name=None):
#     shape = tf.shape(value)
#     pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
#     padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
#     reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
#     transposed = tf.transpose(reshaped, perm=[1, 0, 2])
#     return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])

# a = np.random.rand(3, 15, 100)


# pad = int((3 - 1) * 1 / 2)
# padding = [[0, 0], [pad, pad], [0, 0]]
# padded = tf.pad(a, padding)
# b = time_to_batch(a, 1)

# pad = int((3 - 1) * 2 / 2)
# padding = [[0, 0], [pad, pad], [0, 0]]
# padded = tf.pad(a, padding)
# c = time_to_batch(a, 2)
# print(b.shape)
# print(c.shape)

# # https://github.com/weixsong/WaveGlow/tree/5304ecf811e933c299b4b314c533c2338ecd47dd

# x22050 = librosa.load('D:/Python_Programming/WaveGlow/Wav_for_Inference/FV.SLT.arctic_a0007.wav', sr= 22050)[0]
# x16000 = librosa.core.resample(x22050, 22050, 16000)

# ratio = float(22050 / 16000)
# plt.subplot(411)
# plt.plot(x22050)
# plt.subplot(412)
# plt.plot(x16000)

# plt.subplot(413)
# plt.plot(x22050[int(5000 * ratio):int(21000 * ratio)])
# plt.subplot(414)
# plt.plot(x16000[5000:21000])

# # plt.show()
# from Audio import melspectrogram

# # for x in range(400, 601, 1):
# x22050 = librosa.load('D:/Python_Programming/WaveGlow/Wav_for_Inference/FV.SLT.arctic_a0007.wav', sr= 22050)[0]
# #x16000 = librosa.load('D:/Python_Programming/WaveGlow/Wav_for_Inference/FV.SLT.arctic_a0007.wav', sr= 16000)[0]
# x16000 = librosa.core.resample(x22050, 22050, 16000)
# print(int(256 * 16000/22050), int(1024 * 16000/22050))
# q =melspectrogram(x22050, 513, 256, 1024, 80, 22050, 4)
# q2 =melspectrogram(x16000, 513, int(256 * 16000/22050), int(1024 * 16000/22050), 80, 16000, 4)
# plt.subplot(211)
# plt.imshow(q, aspect='auto', origin='lower')
# plt.subplot(212)
# plt.imshow(q2, aspect='auto', origin='lower')
# plt.show()
# print(q.shape, q2.shape)

import tensorflow as tf

i0 = tf.constant(0)
# m0 = tf.ones([2, 2])
m0 = tf.keras.layers.Input(
    shape= [2],
    dtype= tf.float32
    )

c = lambda i, m: i < 10
b = lambda i, m: [i+1, tf.concat([m, m], axis=0)]
d = tf.while_loop(
    c,
    b,
    loop_vars=[0, m0],
    shape_invariants=[i0.get_shape(), tf.TensorShape([None, 2])])

print(d)

print(tf.range(tf.shape(m0)[1]))