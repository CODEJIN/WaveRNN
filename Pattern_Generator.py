import numpy as np
import json, os, time, pickle, librosa, re, argparse
from concurrent.futures import ThreadPoolExecutor as PE
from collections import deque
from threading import Thread
from random import shuffle

from Audio import melspectrogram, spectrogram, preemphasis, inv_preemphasis

with open('Hyper_Parameters.json', 'r') as f:
    hp_Dict = json.load(f)

using_Extension = [x.upper() for x in ['.wav', '.m4a', '.flac']]

def Pattern_Generate(path, top_db= 60):
    sig = librosa.core.load(
        path,
        sr = hp_Dict['Sound']['Sample_Rate']
        )[0]
    sig = preemphasis(sig)
    sig = librosa.effects.trim(sig, top_db= top_db, frame_length= 32, hop_length= 16)[0] * 0.99
    sig = inv_preemphasis(sig)

    mel = np.transpose(melspectrogram(
        y= sig,
        num_freq= hp_Dict['Sound']['Spectrogram_Dim'],        
        hop_length= hp_Dict['Sound']['Frame_Shift'],
        win_length= hp_Dict['Sound']['Frame_Length'],        
        num_mels= hp_Dict['Sound']['Mel_Dim'],
        sample_rate= hp_Dict['Sound']['Sample_Rate'],
        max_abs_value= hp_Dict['Sound']['Max_Abs_Mel']
        ))

    return sig, mel

def Pattern_File_Generate(path, dataset, file_Prefix='', display_Prefix = '', top_db= 60):
    sig, mel = Pattern_Generate(path, top_db)
    
    new_Pattern_Dict = {
        'Signal': sig.astype(np.float32),
        'Mel': mel.astype(np.float32),
        'Dataset': dataset,
        }

    pickle_File_Name = '{}.{}{}.PICKLE'.format(dataset, file_Prefix, os.path.splitext(os.path.basename(path))[0]).upper()

    with open(os.path.join(hp_Dict['Train']['Pattern_Path'], pickle_File_Name).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Pattern_Dict, f, protocol=2)
            
    print('[{}]'.format(display_Prefix), '{}'.format(path), '->', '{}'.format(pickle_File_Name))


def VCTK_Info_Load(vctk_Path):
    vctk_Wav_Path = os.path.join(vctk_Path, 'wav48').replace('\\', '/')
    with open(os.path.join(vctk_Path, 'VCTK.NonOutlier.txt').replace('\\', '/'), 'r') as f:
        vctk_Non_Outlier_List = [x.strip() for x in f.readlines()]

    vctk_File_Path_List = []
    for root, _, file_Name_List in os.walk(vctk_Wav_Path):
        for file_Name in file_Name_List:
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue
            if not vctk_Non_Outlier_List is None and not file_Name in vctk_Non_Outlier_List:
                continue            
            wav_File_Path = os.path.join(root, file_Name).replace('\\', '/')
            vctk_File_Path_List.append(wav_File_Path)

    print('VCTK info generated: {}'.format(len(vctk_File_Path_List)))
    return vctk_File_Path_List

def LS_Info_Load(ls_Path):
    ls_File_Path_List = []
    for root, _, file_Name_List in os.walk(ls_Path):
        for file_Name in file_Name_List:
            wav_File_Path = os.path.join(root, file_Name).replace('\\', '/')
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue
            ls_File_Path_List.append(wav_File_Path)

    print('LS info generated: {}'.format(len(ls_File_Path_List)))
    return ls_File_Path_List

def TIMIT_Info_Load(timit_Path):
    timit_File_Path_List = []
    for root, _, file_Name_List in os.walk(timit_Path):
        for file_Name in file_Name_List:
            wav_File_Path = os.path.join(root, file_Name).replace('\\', '/')
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue
            timit_File_Path_List.append(wav_File_Path)

    print('TIMIT info generated: {}'.format(len(timit_File_Path_List)))
    return timit_File_Path_List

def LJ_Info_Load(lj_Path):
    lj_File_Path_List = []

    for root, _, file_Name_List in os.walk(lj_Path):
        for file_Name in file_Name_List:
            wav_File_Path = os.path.join(root, file_Name).replace('\\', '/')
            if not os.path.splitext(wav_File_Path)[1].upper() in using_Extension:
                continue
            lj_File_Path_List.append(wav_File_Path)
            
    print('LJ info generated: {}'.format(len(lj_File_Path_List)))
    return lj_File_Path_List

def BC2013_Info_Load(bc2013_Path):
    text_Path_List = []
    for root, _, files in os.walk(bc2013_Path):
        for filename in files:
            if os.path.splitext(filename)[1].upper() != '.txt'.upper():
                continue
            text_Path_List.append(os.path.join(root, filename).replace('\\', '/'))

    bc2013_File_Path_List = []

    for text_Path in text_Path_List:
        wav_Path = text_Path.replace('txt', 'wav')
        if not os.path.exists(wav_Path):
            continue
        bc2013_File_Path_List.append(wav_Path)

    print('BC2013 info generated: {}'.format(len(bc2013_File_Path_List)))
    return bc2013_File_Path_List

def FV_Info_Load(fv_Path):
    text_Path_List = []
    for root, _, file_Name_List in os.walk(fv_Path):
        for file in file_Name_List:
            if os.path.splitext(file)[1] == '.data':
                text_Path_List.append(os.path.join(root, file).replace('\\', '/'))

    fv_File_Path_List = []
    fv_Speaker_Dict = {}
    for text_Path in text_Path_List:        
        speaker = text_Path.split('/')[-3].split('_')[2].upper()
        with open(text_Path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            file_Path, _, _ = line.strip().split('"')

            file_Path = file_Path.strip().split(' ')[1]
            wav_File_Path = os.path.join(
                os.path.split(text_Path)[0].replace('etc', 'wav'),
                '{}.wav'.format(file_Path)
                ).replace('\\', '/')

            fv_File_Path_List.append(wav_File_Path)
            fv_Speaker_Dict[wav_File_Path] = speaker

    print('FV info generated: {}'.format(len(fv_File_Path_List)))
    return fv_File_Path_List, fv_Speaker_Dict


def Metadata_Generate():
    new_Metadata_Dict = {
        'Spectrogram_Dim': hp_Dict['Sound']['Spectrogram_Dim'],
        'Mel_Dim': hp_Dict['Sound']['Mel_Dim'],
        'Frame_Shift': hp_Dict['Sound']['Frame_Shift'],
        'Frame_Length': hp_Dict['Sound']['Frame_Length'],
        'Sample_Rate': hp_Dict['Sound']['Sample_Rate'],
        'Max_Abs_Mel': hp_Dict['Sound']['Max_Abs_Mel'],
        'File_List': [],
        'Sig_Length_Dict': {},
        'Mel_Length_Dict': {},
        'Dataset_Dict': {},
        }

    for root, _, files in os.walk(hp_Dict['Train']['Pattern_Path']):
        for file in files:
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_Dict = pickle.load(f)
                try:
                    new_Metadata_Dict['Sig_Length_Dict'][file] = pattern_Dict['Signal'].shape[0]
                    new_Metadata_Dict['Mel_Length_Dict'][file] = pattern_Dict['Mel'].shape[0]
                    new_Metadata_Dict['Dataset_Dict'][file] = pattern_Dict['Dataset']
                    new_Metadata_Dict['File_List'].append(file)
                except:
                    print('File \'{}\' is not correct pattern file. This file is ignored.'.format(file))

    with open(os.path.join(hp_Dict['Train']['Pattern_Path'], hp_Dict['Train']['Metadata_File'].upper()).replace("\\", "/"), 'wb') as f:
        pickle.dump(new_Metadata_Dict, f, protocol=2)

    print('Metadata generate done.')

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-lj", "--lj_path", required=False)
    argParser.add_argument("-vctk", "--vctk_path", required=False)
    argParser.add_argument("-ls", "--ls_path", required=False)
    argParser.add_argument("-timit", "--timit_path", required=False)
    argParser.add_argument("-bc2013", "--bc2013_path", required=False)
    argParser.add_argument("-fv", "--fv_path", required=False)
    argParser.add_argument("-mc", "--max_count", required=False)
    argParser.add_argument("-mw", "--max_worker", required=False)
    argParser.set_defaults(max_worker = 10)
    argument_Dict = vars(argParser.parse_args())
    
    if not argument_Dict['max_count'] is None:
        argument_Dict['max_count'] = int(argument_Dict['max_count'])

    total_Pattern_Count = 0

    if not argument_Dict['lj_path'] is None:
        lj_File_Path_List = LJ_Info_Load(lj_Path= argument_Dict['lj_path'])
        total_Pattern_Count += len(lj_File_Path_List)
    if not argument_Dict['vctk_path'] is None:
        vctk_File_Path_List = VCTK_Info_Load(vctk_Path= argument_Dict['vctk_path'])
        total_Pattern_Count += len(vctk_File_Path_List)
    if not argument_Dict['ls_path'] is None:
        ls_File_Path_List = LS_Info_Load(ls_Path= argument_Dict['ls_path'])
        total_Pattern_Count += len(ls_File_Path_List)
    if not argument_Dict['timit_path'] is None:
        timit_File_Path_List = TIMIT_Info_Load(timit_Path= argument_Dict['timit_path'])
        total_Pattern_Count += len(timit_File_Path_List)
    if not argument_Dict['bc2013_path'] is None:
        bc2013_File_Path_List = BC2013_Info_Load(bc2013_Path= argument_Dict['bc2013_path'])
        total_Pattern_Count += len(bc2013_File_Path_List)
    if not argument_Dict['fv_path'] is None:
        fv_File_Path_List, fv_Speaker_Dict = FV_Info_Load(fv_Path= argument_Dict['fv_path'])
        total_Pattern_Count += len(fv_File_Path_List)

    if total_Pattern_Count == 0:
        raise ValueError('Total pattern count is zero.')
    
    os.makedirs(hp_Dict['Train']['Pattern_Path'], exist_ok= True)
    total_Generated_Pattern_Count = 0
    with PE(max_workers = int(argument_Dict['max_worker'])) as pe:
        if not argument_Dict['lj_path'] is None:            
            for index, file_Path in enumerate(lj_File_Path_List):
                pe.submit(
                    Pattern_File_Generate,
                    file_Path,
                    'LJ',
                    '',
                    'LJ {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                        index,
                        len(lj_File_Path_List),
                        total_Generated_Pattern_Count,
                        total_Pattern_Count
                        ),
                    60
                    )
                total_Generated_Pattern_Count += 1

        if not argument_Dict['vctk_path'] is None:
            for index, file_Path in enumerate(vctk_File_Path_List):
                pe.submit(
                    Pattern_File_Generate,
                    file_Path,
                    'VCTK',
                    '',
                    'VCTK {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                        index,
                        len(vctk_File_Path_List),
                        total_Generated_Pattern_Count,
                        total_Pattern_Count
                        ),
                    15
                    )
                total_Generated_Pattern_Count += 1

        if not argument_Dict['ls_path'] is None:
            for index, file_Path in enumerate(ls_File_Path_List):
                pe.submit(
                    Pattern_File_Generate,
                    file_Path,
                    'LS',
                    '',
                    'LS {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                        index,
                        len(ls_File_Path_List),
                        total_Generated_Pattern_Count,
                        total_Pattern_Count
                        ),
                    60
                    )
                total_Generated_Pattern_Count += 1

        if not argument_Dict['timit_path'] is None:
            for index, file_Path in enumerate(timit_File_Path_List):
                pe.submit(
                    Pattern_File_Generate,
                    file_Path,
                    'TIMIT',
                    '{}.'.format(file_Path.split('/')[-2]),
                    'TIMIT {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                        index,
                        len(timit_File_Path_List),
                        total_Generated_Pattern_Count,
                        total_Pattern_Count
                        ),
                    60
                    )
                total_Generated_Pattern_Count += 1

        if not argument_Dict['bc2013_path'] is None:
            for index, file_Path in enumerate(bc2013_File_Path_List):
                pe.submit(
                    Pattern_File_Generate,
                    file_Path,
                    'BC2013',
                    '{}.'.format(file_Path.split('/')[-2]),
                    'BC2013 {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                        index,
                        len(bc2013_File_Path_List),
                        total_Generated_Pattern_Count,
                        total_Pattern_Count
                        ),
                    60
                    )
                total_Generated_Pattern_Count += 1

        if not argument_Dict['fv_path'] is None:
            for index, file_Path in enumerate(fv_File_Path_List):
                pe.submit(
                    Pattern_File_Generate,
                    file_Path,
                    'FV',
                    '{}.'.format(fv_Speaker_Dict[file_Path]),
                    'FV {:05d}/{:05d}    Total {:05d}/{:05d}'.format(
                        index,
                        len(fv_File_Path_List),
                        total_Generated_Pattern_Count,
                        total_Pattern_Count
                        ),
                    60
                    )
                total_Generated_Pattern_Count += 1

    Metadata_Generate()