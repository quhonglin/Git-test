import os
import random

import librosa
import numpy as np
import torchaudio as torchaudio
from joblib import Parallel, delayed
from torch.utils import data

from util.utils import sample_fixed_length_data_aligned


class Dataset(data.Dataset):
    def __init__(self, dataset_list, limit=None, offset=0, sr=16000, n_samples=32000):
        """
        训练和验证数据集

        dataset_list(*.txt):
            <mixture_path> <target_path> \n
        e.g:
            mixture_1.wav target_1.wav
            mixture_2.wav target_2.wav
            ...
            mixture_n.wav target_n.wav
        """
        super(Dataset, self).__init__()
        dataset_list = [line.rstrip('\n') for line in
                        open(os.path.abspath(os.path.expanduser(dataset_list)), "r")]

        dataset_list = dataset_list[offset:]
        if limit:
            dataset_list = dataset_list[:limit]

        self.dataset_list = dataset_list
        self.length = len(self.dataset_list)
        self.sr = sr
        self.n_samples = n_samples

    @staticmethod
    def get_filename(file_path):
        filename = os.path.splitext(os.path.basename(file_path))[0]
        return filename

    @staticmethod
    def get_speaker_id(filename):
        speaker_id = filename.split("_")[1]
        return speaker_id

    def __len__(self):
        return self.length

    def load_wav(self, file_path):
        return librosa.load(os.path.abspath(os.path.expanduser(file_path)), sr=self.sr)[0]

    def __getitem__(self, item):
        mixture_path, target_path, _ = self.dataset_list[item].split(" ")

        target_filename = self.get_filename(target_path)

        mixture_y = self.load_wav(mixture_path)
        target_y = self.load_wav(target_path)

        mixture_y, target_y = sample_fixed_length_data_aligned(mixture_y, target_y, self.n_samples)

        return mixture_y.astype(np.float32), target_y.astype(np.float32), target_filename
