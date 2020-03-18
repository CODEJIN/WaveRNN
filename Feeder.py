import numpy as np
import json, os, time, librosa, pickle
from collections import deque
from threading import Thread
from random import shuffle, randint
from Audio import melspectrogram, spectrogram, preemphasis, inv_preemphasis
from Pattern_Generator import Pattern_Generate as Load_Mel_from_Signal

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

class Feeder:
    def __init__(self, is_Training= False):
        self.is_Training = is_Training

        if self.is_Training:
            if hp_Dict['Train']['Wav_Length'] % hp_Dict['Sound']['Frame_Shift'] != 0:
                raise ValueError('The wav length of train must be a multiple of frame shift size.')

            self.Metadata_Load()

            self.signal_Dict = {}

            self.pattern_Queue = deque()
            pattern_Generate_Thread = Thread(target= self.Pattern_Generate)
            pattern_Generate_Thread.daemon = True
            pattern_Generate_Thread.start()

    def Metadata_Load(self):
        if self.is_Training:
            with open(os.path.join(hp_Dict['Train']['Pattern_Path'], hp_Dict['Train']['Metadata_File']).replace('\\', '/'), 'rb') as f:
                self.metadata_Dict = pickle.load(f)

            if not all([
                self.metadata_Dict['Spectrogram_Dim'] == hp_Dict['Sound']['Spectrogram_Dim'],
                self.metadata_Dict['Mel_Dim'] == hp_Dict['Sound']['Mel_Dim'],
                self.metadata_Dict['Frame_Shift'] == hp_Dict['Sound']['Frame_Shift'],
                self.metadata_Dict['Frame_Length'] == hp_Dict['Sound']['Frame_Length'],
                self.metadata_Dict['Sample_Rate'] == hp_Dict['Sound']['Sample_Rate'],
                self.metadata_Dict['Max_Abs_Mel'] == hp_Dict['Sound']['Max_Abs_Mel'],
                ]):
                raise ValueError('The metadata information and hyper parameter setting are not consistent.')

    def Pattern_Generate(self):
        path_List = [path for path in self.metadata_Dict['File_List']]

        print(
            'Train pattern info', '\n',
            'Pattern count: {}'.format(len(path_List))
            )

        mel_Window = hp_Dict['Train']['Wav_Length'] // hp_Dict['Sound']['Frame_Shift'] + 2 * hp_Dict['WaveRNN']['Upsample']['Pad']
        while True:
            shuffle(path_List)            
            path_Batch_List = [
                path_List[x:x+hp_Dict['Train']['Batch_Size']]
                for x in range(0, len(path_List), hp_Dict['Train']['Batch_Size'])
                ]
            shuffle(path_Batch_List)

            batch_Index = 0
            while batch_Index < len(path_Batch_List):
                if len(self.pattern_Queue) >= hp_Dict['Train']['Max_Pattern_Queue']:
                    time.sleep(0.1)
                    continue
                
                mel_List = []
                sig_List = []
                for path in path_Batch_List[batch_Index]:
                    with open(os.path.join(hp_Dict['Train']['Pattern_Path'], path).replace('\\', '/'), 'rb') as f:
                        pattern_Dict = pickle.load(f)

                    if pattern_Dict['Signal'].shape[0] < mel_Window:
                        continue

                    max_Offset = pattern_Dict['Mel'].shape[0] - 2 - (mel_Window + 2 * hp_Dict['WaveRNN']['Upsample']['Pad'])
                    mel_Offset = np.random.randint(0, max_Offset)
                    sig_Offset = (mel_Offset + hp_Dict['WaveRNN']['Upsample']['Pad']) * hp_Dict['Sound']['Frame_Shift']
                    
                    mel_List.append(pattern_Dict['Mel'][mel_Offset:mel_Offset + mel_Window])
                    sig_List.append(pattern_Dict['Signal'][sig_Offset:sig_Offset + hp_Dict['Train']['Wav_Length'] + 1])  # +1 is for input and target difference
                    
                sig_Pattern = np.stack(sig_List, axis= 0)
                mel_Pattern = np.stack(mel_List, axis= 0)

                self.pattern_Queue.append({
                    'audios': sig_Pattern,
                    'mels': mel_Pattern
                    })
        
    def Get_Pattern(self):
        while len(self.pattern_Queue) == 0: #When training speed is faster than making pattern, model should be wait.
            time.sleep(0.01)
        return self.pattern_Queue.popleft()

    # def Get_Inference_Pattern(self, mel_List= None, wav_List= None):
    #     if wav_List is None and mel_List is None:
    #         print('One of paths must be not None.')
    #         return None
    #     sig_List = []
    #     mel_List = mel_List or []
    #     for path in wav_List:
    #         sig, mel = Load_Mel_from_Signal(path)
    #         sig_List.append(sig)
    #         mel_List.append(mel)

    #     max_Mel_Length = max([mel.shape[0] for mel in mel_List])
    #     mel_Pattern = np.zeros(
    #         shape= (len(mel_List), max_Mel_Length, hp_Dict['Sound']['Mel_Dim']),
    #         dtype= np.float32
    #         )
    #     for index, mel in enumerate(mel_List):
    #         mel_Pattern[index, :mel.shape[0]] = mel

    #     wav_List = [None] * len(mel_List)
    #     wav_List[-len(sig_List):] = sig_List
    #     pattern_Dict = {
    #         'audios': np.zeros(shape=(len(mel_List), 1), dtype= np.float32),
    #         'mels': mel_Pattern,
    #         }

    #     return wav_List, pattern_Dict

    def Get_Inference_Pattern(
        self,
        mel_List= None,
        wav_List= None,
        split_Mel_Window= 7,
        overlap_Window= 1,
        batch_Size= 16
        ):
        split_Mel_Window += 2 * hp_Dict['WaveRNN']['Upsample']['Pad']
        overlap_Window += 2 * hp_Dict['WaveRNN']['Upsample']['Pad']

        if wav_List is None and mel_List is None:
            print('One of paths must be not None.')
            return None, None, None
        sig_List = []
        mel_List = mel_List or []
        for path in wav_List:
            sig, mel = Load_Mel_from_Signal(path)
            sig_List.append(sig)
            mel_List.append(mel)

        split_Mel_Index_List = []
        split_Mel_List = []
        for index, mel in enumerate(mel_List):
            mel = np.vstack([np.zeros(shape=(overlap_Window, mel.shape[1]), dtype= mel.dtype), mel])    # initial padding
            current_Index = 0
            while True:
                split_Mel_Index_List.append(index)
                split_Mel_List.append(mel[current_Index:current_Index + split_Mel_Window])
                
                if current_Index + split_Mel_Window >= mel.shape[0]:
                    break
                current_Index += split_Mel_Window - overlap_Window                
            split_Mel_List[-1] = np.vstack([
                split_Mel_List[-1],
                np.zeros(shape=(split_Mel_Window - split_Mel_List[-1].shape[0], mel.shape[1]), dtype= mel.dtype)
                ])    # last padding

        mel_Pattern = np.stack(split_Mel_List, axis= 0)

        wav_List = [None] * len(mel_List)
        wav_List[-len(sig_List):] = sig_List

        pattern_Dict_List = []
        for split_Mel_Pattern in [mel_Pattern[index:index+batch_Size] for index in range(0, mel_Pattern.shape[0], batch_Size)]:
            new_Pattern_Dict = {
                'audios': np.zeros(
                    shape=(
                        split_Mel_Pattern.shape[0],
                        hp_Dict['Sound']['Frame_Shift'] * (split_Mel_Window - 2 * hp_Dict['WaveRNN']['Upsample']['Pad'])
                        ),
                    dtype= np.float32
                    ),
                'mels': split_Mel_Pattern,
                }
            pattern_Dict_List.append(new_Pattern_Dict)

        return wav_List, pattern_Dict_List, split_Mel_Index_List


if __name__ == "__main__":
    new_Feeder = Feeder(is_Training= True)

    print(new_Feeder.Get_Inference_Pattern(wav_List= [
        'D:/Pattern/ENG/LJSpeech/wavs/LJ050-0276.wav',
        'D:/Pattern/ENG/FastVox/cmu_us_jmk_arctic/wav/arctic_a0012.wav',
        ]))
    while True:
        time.sleep(1.0)
        print(new_Feeder.Get_Pattern())
        assert False