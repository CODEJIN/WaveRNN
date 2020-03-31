# Refer:
# https://https://github.com/fatchord/WaveRNN

import tensorflow as tf
import numpy as np
import json
from MoL import Sample_from_Discretized_Mix_Logistic, Discretized_Mix_Logistic_Loss

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

class ResBlock(tf.keras.layers.Layer):
    def build(self, input_shapes):
        self.layer = tf.keras.Sequential()
        self.layer.add(tf.keras.layers.Conv1D(
            filters= input_shapes[-1],
            kernel_size= 1,
            strides= 1,
            use_bias= False
            ))
        self.layer.add(tf.keras.layers.BatchNormalization())
        self.layer.add(tf.keras.layers.ReLU())
        self.layer.add(tf.keras.layers.Conv1D(
            filters= input_shapes[-1],
            kernel_size= 1,
            strides= 1,
            use_bias= False
            ))
        self.layer.add(tf.keras.layers.BatchNormalization())

    def call(self, inputs, training):
        return self.layer(inputs, training) + inputs

class MelResNet(tf.keras.Model):
    def __init__(
        self,
        res_blocks,
        compute_dims,
        output_dims,
        pad #kernel size: pad * 2 + 1
        ):
        super(MelResNet, self).__init__()
        self.res_blocks = res_blocks
        self.compute_dims = compute_dims
        self.output_dims = output_dims
        self.pad = pad

    def build(self, input_shapes):
        self.layer = tf.keras.Sequential()
        self.layer.add(tf.keras.layers.Conv1D(
            filters= self.compute_dims,
            kernel_size= self.pad * 2 + 1,
            strides= 1,
            use_bias= False
            ))  #valid
        self.layer.add(tf.keras.layers.BatchNormalization())
        self.layer.add(tf.keras.layers.ReLU())
        for _ in range(self.res_blocks):
            self.layer.add(ResBlock())
        self.layer.add(tf.keras.layers.Conv1D(
            filters= self.output_dims,
            kernel_size= 1,
            strides= 1
            ))  #valid

    def call(self, inputs, training):
        return self.layer(inputs, training)

class UpsampleNet(tf.keras.Model):
    def __init__(
        self,
        res_blocks,
        upsample_scales,
        compute_dims,
        output_dims,
        pad
        ):
        super(UpsampleNet, self).__init__()
        self.res_blocks = res_blocks
        self.upsample_scales = upsample_scales
        self.compute_dims = compute_dims
        self.output_dims = output_dims
        self.pad = pad

        self.total_scale = np.cumproduct(self.upsample_scales)[-1]
        self.indent = self.pad * self.total_scale

    def build(self, input_shapes):
        self.layer_Dict = {}

        self.layer_Dict['Aux'] = tf.keras.Sequential()
        self.layer_Dict['Aux'].add(MelResNet(
            res_blocks= self.res_blocks,
            compute_dims= self.compute_dims,
            output_dims= self.output_dims,
            pad= self.pad
            ))
        self.layer_Dict['Aux'].add(tf.keras.layers.UpSampling1D(size= self.total_scale))
        
        self.layer_Dict['Mel'] = tf.keras.Sequential()
        self.layer_Dict['Mel'].add(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis= -1)))
        for scale in self.upsample_scales:
            self.layer_Dict['Mel'].add(tf.keras.layers.UpSampling2D(size= (scale, 1)))
            self.layer_Dict['Mel'].add(tf.keras.layers.Conv2D(
                filters= 1,
                kernel_size= (scale * 2 + 1, 1),
                kernel_initializer= tf.constant_initializer(1 / (scale * 2 + 1)),
                padding= 'same',
                use_bias= False
                ))
        self.layer_Dict['Mel'].add(tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis= -1)))
        self.built = True

    def call(self, inputs, training):
        mel_Tensor = self.layer_Dict['Mel'](inputs, training)
        mel_Tensor = mel_Tensor[:, self.indent:-self.indent, :]
        aux_Tensor = self.layer_Dict['Aux'](inputs, training)
        
        return mel_Tensor, aux_Tensor

class WaveRNN(tf.keras.Model):
    def build(self, input_shapes):
        self.layer_Dict = {}

        if np.cumproduct(hp_Dict['WaveRNN']['Upsample']['Scales'])[-1] != hp_Dict['Sound']['Frame_Shift']:
            raise ValueError('The product of all of upsample scales must be same to frame shift size.')

        self.layer_Dict['Upsample'] = UpsampleNet(
            res_blocks= hp_Dict['WaveRNN']['Upsample']['Res_Blocks'],
            upsample_scales= hp_Dict['WaveRNN']['Upsample']['Scales'],
            compute_dims= hp_Dict['WaveRNN']['Upsample']['Hidden_Size'],
            output_dims= hp_Dict['WaveRNN']['Upsample']['Output_Size'],
            pad= hp_Dict['WaveRNN']['Upsample']['Pad']
            )

        self.layer_Dict['I'] = tf.keras.layers.Dense(
            units= hp_Dict['WaveRNN']['RNN_Size']
            )

        self.layer_Dict['RNN_Cell_0'] = tf.keras.layers.GRUCell(
            units= hp_Dict['WaveRNN']['RNN_Size']
            )
        self.layer_Dict['RNN_0'] = tf.keras.layers.RNN(
            cell= self.layer_Dict['RNN_Cell_0'],
            return_sequences= True
            )
        self.layer_Dict['RNN_Cell_1'] = tf.keras.layers.GRUCell(
            units= hp_Dict['WaveRNN']['RNN_Size']
            )
        self.layer_Dict['RNN_1'] = tf.keras.layers.RNN(
            cell= self.layer_Dict['RNN_Cell_1'],
            return_sequences= True
            )

        self.layer_Dict['Dense_0'] = tf.keras.layers.Dense(
            units= hp_Dict['WaveRNN']['RNN_Size'],
            activation= 'relu'
            )
        self.layer_Dict['Dense_1'] = tf.keras.layers.Dense(
            units= hp_Dict['WaveRNN']['RNN_Size'],
            activation= 'relu'
            )

        if hp_Dict['WaveRNN']['Mode'].upper() == 'MoL'.upper():
            projection_Size = 30
        elif hp_Dict['WaveRNN']['Mode'].upper() == 'Raw'.upper():
            projection_Size = 512   #Coarse + Fine
        else:
            raise ValueError('Unsupported mode')

        self.layer_Dict['Projection'] = tf.keras.layers.Dense(
            units= projection_Size
            )

    def call(self, inputs, training= False):
        '''
        inputs: audios, mels
        audios: [Batch, Time + 1] when train | [Batch, 1] when inference
        mels: [Batch, Time, Mel_dim]
        '''
        audios, mels = inputs
        training = tf.convert_to_tensor(training)
        mels, auxs = self.layer_Dict['Upsample'](mels)

        logits, samples = tf.cond(
            pred= training,
            true_fn= lambda: self.train(audios, mels, auxs),
            false_fn= lambda: self.inference(mels, auxs)
            )

        return logits, samples

    def train(self, audios, mels, auxs):
        audios = tf.expand_dims(audios, axis= -1)
        auxs = tf.split(auxs, num_or_size_splits= 4, axis= -1)

        new_Tensor = tf.concat([audios, mels, auxs[0]], axis= -1)
        new_Tensor = self.layer_Dict['I'](new_Tensor)
        new_Tensor += self.layer_Dict['RNN_0'](new_Tensor)
        new_Tensor += self.layer_Dict['RNN_1'](tf.concat([new_Tensor, auxs[1]], axis= -1))
        new_Tensor = self.layer_Dict['Dense_0'](tf.concat([new_Tensor, auxs[2]], axis= -1))
        new_Tensor = self.layer_Dict['Dense_1'](tf.concat([new_Tensor, auxs[3]], axis= -1))
        logits = self.layer_Dict['Projection'](new_Tensor)

        return logits, tf.zeros(shape=[tf.shape(logits)[0], tf.shape(logits)[1]], dtype= mels.dtype)

    def inference(self, mels, auxs):
        batch_Size, mel_Lengths = tf.shape(mels)[0], tf.shape(mels)[1]

        initial_Hidden_0 = self.layer_Dict['RNN_Cell_0'].get_initial_state(inputs= mels)
        initial_Hidden_1 = self.layer_Dict['RNN_Cell_1'].get_initial_state(inputs= mels)
        samples = tf.zeros(shape=[batch_Size, 1], dtype= mels.dtype)    # [Batch, 1] for inference
        def body(step, hidden_0, hidden_1, samples):
            current_Mel = mels[:, step, :]
            current_Aux = tf.split(auxs[:, step, :], num_or_size_splits= 4, axis= -1)
            current_Sample = tf.expand_dims(samples[:, -1], axis= -1)

            new_Tensor = tf.concat([current_Sample, current_Mel, current_Aux[0]], axis= -1)
            new_Tensor = self.layer_Dict['I'](new_Tensor)            
            new_Hidden_0, _ = self.layer_Dict['RNN_Cell_0'](
                inputs= new_Tensor,
                # states= tf.expand_dims(hidden_0, axis= 0)   #Problem: https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/keras/layers/recurrent.py#L1770
                states= hidden_0    # TF 2.2.0rc
                )

            new_Tensor = new_Hidden_0 + new_Tensor
            new_Hidden_1, _ = self.layer_Dict['RNN_Cell_1'](
                inputs= tf.concat([new_Tensor, current_Aux[1]], axis= -1),   #There is no fc, so there is no residual.
                #states= tf.expand_dims(hidden_1, axis= 0)    #Problem: https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/keras/layers/recurrent.py#L1770
                states= hidden_1    # TF 2.2.0rc
                )

            new_Tensor = new_Hidden_1 + new_Tensor
            new_Tensor = self.layer_Dict['Dense_0'](
                tf.concat([new_Tensor, current_Aux[2]], axis= -1)
                )
            new_Tensor = self.layer_Dict['Dense_1'](
                tf.concat([new_Tensor, current_Aux[3]], axis= -1)
                )
            logit = self.layer_Dict['Projection'](new_Tensor)

            sample = self.calc_sample(logit)
            samples = tf.concat([samples, sample], axis= -1)
        
            return step + 1, new_Hidden_0, new_Hidden_1, samples
        
        _, _, _, samples = tf.while_loop(
            cond= lambda step, hidden_0, hidden_1, samples: step < mel_Lengths,
            body= body,
            loop_vars= [
                0,
                initial_Hidden_0,
                initial_Hidden_1,
                samples
                ],
            shape_invariants= [
                tf.TensorShape([]),
                tf.TensorShape([None, initial_Hidden_0.get_shape()[-1]]),
                tf.TensorShape([None, initial_Hidden_1.get_shape()[-1]]),
                tf.TensorShape([None, None]),
                ]
            )

        return tf.zeros(shape=[batch_Size, mel_Lengths, self.layer_Dict['Projection'].units], dtype= mels.dtype), samples[:, 1:] # [Batch, Time], initial zero slice


    def calc_sample(self, logit):
        if hp_Dict['WaveRNN']['Mode'].upper() == 'MoL'.upper():
            sample = Sample_from_Discretized_Mix_Logistic(logit)
            sample = tf.expand_dims(sample, axis= -1)
        elif hp_Dict['WaveRNN']['Mode'].upper() == 'Raw'.upper():
            coarse_Logit, fine_Logit = tf.split(logit, num_or_size_splits= 2, axis= -1)                
            coarse = tf.random.categorical(tf.math.softmax(coarse_Logit, axis= -1), num_samples= 1)
            fine = tf.random.categorical(tf.math.softmax(fine_Logit, axis= -1), num_samples= 1)
            sample = 2 * (coarse * 256 + fine) / 65536 - 1.0
            sample = tf.cast(sample, dtype= logit.dtype)
        else:
            raise ValueError('Unsupported mode')
        return sample
        

class Loss(tf.keras.layers.Layer):
    def call(self, inputs):        
        labels, logits = inputs
        if hp_Dict['WaveRNN']['Mode'].upper() == 'MoL'.upper():
            return Discretized_Mix_Logistic_Loss(labels= labels, logits= logits)
        elif hp_Dict['WaveRNN']['Mode'].upper() == 'Raw'.upper():
            labels = tf.cast((labels + 1) / 2 * (65536 - 1), dtype=tf.int32)
            coarse_Labels, fine_Labels = labels // 256, labels % 256            
            coarse_Logits, fine_Logits = tf.split(logits, num_or_size_splits= 2, axis= -1)

            coarse_Loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels= coarse_Labels,
                logits= coarse_Logits
                ))
            fine_Loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels= fine_Labels,
                logits= fine_Logits
                ))
            return coarse_Loss + fine_Loss
        else:
            raise ValueError('Unsupported mode')


class ExponentialDecay(tf.keras.optimizers.schedules.ExponentialDecay):
    def __init__(
        self,
        initial_learning_rate,
        decay_steps,
        decay_rate,
        min_learning_rate= None,
        staircase=False,
        name=None
        ):    
        super(ExponentialDecay, self).__init__(
            initial_learning_rate= initial_learning_rate,
            decay_steps= decay_steps,
            decay_rate= decay_rate,
            staircase= staircase,
            name= name
            )

        self.min_learning_rate = min_learning_rate

    def __call__(self, step):
        learning_rate = super(ExponentialDecay, self).__call__(step)
        if self.min_learning_rate is None:
            return learning_rate

        return tf.maximum(learning_rate, self.min_learning_rate)

    def get_config(self):
        config_dict = super(ExponentialDecay, self).get_config()
        config_dict['min_learning_rate'] = self.min_learning_rate

        return config_dict


if __name__ == "__main__":
    up = UpsampleNet(10, [4,4,16], 128, 128, 2)

    import librosa
    sig = librosa.load('D:/Python_Programming/WaveRNN/Wav_for_Inference/FV.AWB.arctic_a0001.wav', sr= 16000)[0]
    
    sig = sig[500:500+(256*5)]
    mel = melspectrogram(sig, 513, 80, 256, 1024, 16000, 4)

    print(sig.shape)
    print(mel.shape)
    up_mels = up(tf.expand_dims(mel, axis=0))

    print(up_mels[0].shape)
    print(up_mels[1].shape)
    