
# class Reconstruct(tf.keras.layers.Layer):
#     def call(self, inputs):
#         sig = inputs

#         unsigned_sig = tf.cast((sig + 1) * 2 ** 15, dtype= tf.int32)    # loss function does not support uint16

#         coarse = unsigned_sig // 2 ** 8
#         fine = unsigned_sig % 2 ** 8
#         mel = melspectrogram(
#             signals= sig,
#             num_freq= hp_Dict['Sound']['Spectrogram_Dim'],
#             num_mels= hp_Dict['Sound']['Mel_Dim'],
#             hop_length= hp_Dict['Sound']['Frame_Shift'],
#             win_length= hp_Dict['Sound']['Frame_Length'],
#             sample_rate= hp_Dict['Sound']['Sample_Rate'],
#             max_abs_value= hp_Dict['Sound']['Max_Abs_Mel'],
#             )

#         print(coarse.shape)
#         print(fine.shape)

#         return coarse, fine, mel

class Reconstruct(tf.keras.layers.Layer):
    def build(self, input_shapes):
        self.mel_upsample_layer = Conv1DTranspose(
            filters= hp_Dict['Sound']['Mel_Dim'],
            kernel_size= hp_Dict['Sound']['Frame_Length'],
            strides= hp_Dict['Sound']['Frame_Shift']
            )

        self.built = True

    def call(self, inputs):
        sig = inputs

        unsigned_sig = tf.cast((sig + 1) * 2 ** 15, dtype= tf.int32)    # loss function does not support uint16

        coarse = unsigned_sig // 2 ** 8
        fine = unsigned_sig % 2 ** 8
        mel = melspectrogram(
            signals= sig,
            num_freq= hp_Dict['Sound']['Spectrogram_Dim'],
            num_mels= hp_Dict['Sound']['Mel_Dim'],
            hop_length= hp_Dict['Sound']['Frame_Shift'],
            win_length= hp_Dict['Sound']['Frame_Length'],
            sample_rate= hp_Dict['Sound']['Sample_Rate'],
            max_abs_value= hp_Dict['Sound']['Max_Abs_Mel'],
            )
        mel = self.mel_upsample_layer(mel)
        mel = mel[:, :sig.get_shape()[-1], :]

        return coarse, fine, mel

class WaveRNN(tf.keras.Model):
    def __init__(self):
        super(WaveRNN, self).__init__()

    def build(self, input_shapes):
        if hp_Dict['WaveRNN']['RNN_Size'] % 2 != 0:
            raise ValueError('RNN size must be even number.')

        self.layer_Dict = {}
        self.layer_Dict['RNN_Cell'] = tf.keras.layers.GRUCell(
            units= hp_Dict['WaveRNN']['RNN_Size']
            )
        self.layer_Dict['RNN'] = tf.keras.layers.RNN(
            self.layer_Dict['RNN_Cell'],
            return_sequences= True
            )
        self.layer_Dict['Coarse'] = tf.keras.Sequential()
        self.layer_Dict['Coarse'].add(tf.keras.layers.Dense(
            units= hp_Dict['WaveRNN']['RNN_Size'] // 2,
            activation= 'relu'
            ))
        self.layer_Dict['Coarse'].add(tf.keras.layers.Dense(
            units= hp_Dict['WaveRNN']['Class']
            ))

        self.layer_Dict['Fine'] = tf.keras.Sequential()
        self.layer_Dict['Fine'].add(tf.keras.layers.Dense(
            units= hp_Dict['WaveRNN']['RNN_Size'] // 2,
            activation= 'relu'
            ))
        self.layer_Dict['Fine'].add(tf.keras.layers.Dense(
            units= hp_Dict['WaveRNN']['Class']
            ))

        self.built = True

    def train(self, inputs):
        '''
        inputs: coarses, fines, mels
        coarses: [Batch, Sig_t]
        fines: [Batch, Sig_t]
        mels: [Batch, Mel_t, Mel_dim]
        '''
        if not self.built:            
            self.build(tf.nest.map_structure(lambda x: x.shape, inputs))

        print('##################################################')
        coarses, fines, mels = inputs

        coarses_Val = tf.expand_dims(tf.cast(coarses, dtype= mels.dtype) / 255.0 * 2 - 1, axis= -1)
        fines_Val = tf.expand_dims(tf.cast(fines, dtype= mels.dtype) / 255.0 * 2 - 1, axis= -1)

        x = tf.concat([mels[:, :-1], coarses_Val[:, :-1], fines_Val[:, :-1], coarses_Val[:, 1:]], axis= -1)
        h = self.layer_Dict['RNN'](x)

        h_coarse, h_fine = tf.split(h, num_or_size_splits= 2, axis= -1)
        p_coarse = self.layer_Dict['Coarse'](h_coarse)
        p_fine = self.layer_Dict['Fine'](h_fine)

        return p_coarse, p_fine

    def call(self, inputs):
        # if not self.built:            
        #     self.build(tf.nest.map_structure(lambda x: x.shape, inputs))

        def Calc_Coarse(x, mel, hidden):
            # Because of 'https://github.com/tensorflow/tensorflow/blob/e5bf8de410005de06a7ff5393fafdf832ef1d4ad/tensorflow/python/keras/layers/recurrent.py#L1770'
            h_0, _ = self.layer_Dict['RNN_Cell'](
                inputs= tf.concat([mel, x], axis= -1),
                states= tf.expand_dims(hidden, axis = 0)
                )

            h_Coarse, _ = tf.split(h_0, num_or_size_splits= 2, axis= -1)
            return self.layer_Dict['Coarse'](h_Coarse)

        def Calc_Fine(x, mel, hidden):
            h_1, _ = self.layer_Dict['RNN_Cell'](
                inputs= tf.concat([mel, x], axis= -1),
                states= tf.expand_dims(hidden, axis = 0)
                )
            _, h_Fine = tf.split(h_1, num_or_size_splits= 2, axis= -1)
            return self.layer_Dict['Fine'](h_Fine), h_1

        mels = inputs
        batch_Size = tf.shape(mels)[0]
        initial_Coarse_Val = tf.zeros((batch_Size, 1), dtype= mels.dtype, name='initial_coarse_val')
        initial_Fine_Val = tf.zeros((batch_Size, 1), dtype= mels.dtype, name='initial_fine_val')
        initial_Hidden = self.layer_Dict['RNN_Cell'].get_initial_state(inputs= mels)
        samples = tf.zeros(
            shape=[batch_Size, 1],
            dtype= mels.dtype,
            name='initial_samples'
            )    # [Batch, 1]

        dummy_Coarse = tf.zeros((batch_Size, 1), dtype= mels.dtype, name='dummy_coarse')

        def body(step, coarse, fine, hidden, samples):
            current_Mel = mels[:, step, :]
            x = tf.concat([coarse, fine, dummy_Coarse], axis= -1)
            new_Coarse = Calc_Coarse(x= x, mel= current_Mel, hidden= hidden)
            new_Coarse = tf.math.softmax(new_Coarse)
            new_Coarse = tf.random.categorical(new_Coarse, num_samples= 1)
            new_Coarse = tf.cast(new_Coarse, dtype= mels.dtype)
            new_Coarse_Val = new_Coarse / 255 * 2 - 1.0

            x = tf.concat([coarse, fine, new_Coarse_Val], axis= -1)
            new_Fine, hidden = Calc_Fine(x= x, mel= current_Mel, hidden= hidden)
            new_Fine = tf.math.softmax(new_Fine)
            new_Fine = tf.random.categorical(new_Fine, num_samples= 1)
            new_Fine = tf.cast(new_Fine, dtype= mels.dtype)
            fine = new_Fine / 255 * 2 - 1.0

            coarse = new_Coarse_Val

            sample = (new_Coarse * hp_Dict['WaveRNN']['Class'] + new_Fine) / 32767.5 - 1.0            
            samples = tf.concat([samples, sample], axis= -1)
        
            return step + 1, coarse, fine, hidden, samples
        
        _, _, _, _, samples = tf.while_loop(
            cond= lambda step, coarse_Val, fine_Val, hidden, samples: tf.less(step, tf.shape(mels)[1]), #Why does not work this?
            body= body,
            loop_vars= [
                0,
                initial_Coarse_Val,
                initial_Fine_Val,
                initial_Hidden,
                samples
                ],
            shape_invariants= [
                tf.TensorShape([]),
                tf.TensorShape([None, 1]),
                tf.TensorShape([None, 1]),
                tf.TensorShape([None, hp_Dict['WaveRNN']['RNN_Size']]),
                tf.TensorShape([None, None]),
                ]
            )

        return samples[:, 1:]    #Delete initial zeros

class Loss(tf.keras.layers.Layer):
    def call(self, inputs):
        coarses, fines, p_coarse, p_fine = inputs
        coarse_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels= coarses[:, 1:],
            logits= p_coarse
            ))
        fine_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels= fines[:, 1:],
            logits= p_fine
            ))

        return coarse_loss + fine_loss


class Conv1DTranspose(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides):
        super(Conv1DTranspose, self).__init__()
        self.filters= filters
        self.kernel_size = kernel_size
        self.strides = strides

    def build(self, input_shapes):
        self.layer = tf.keras.layers.Conv2DTranspose(
            filters= self.filters,
            kernel_size = (1, self.kernel_size),
            strides = (1, self.strides),
            )
    def call(self, inputs):
        new_tensor = tf.expand_dims(inputs, axis= 1)
        new_tensor = self.layer(new_tensor)
        return tf.squeeze(new_tensor, axis= 1)
