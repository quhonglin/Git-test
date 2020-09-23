import os
import librosa
import librosa.display
from util.utils import sample_fixed_length_data_aligned
import numpy as np
from util import visualization
from tqdm import tqdm


def load_wav(file_path):
    return librosa.load(os.path.abspath(os.path.expanduser(file_path)), sr=16000)[0]

sr = 16000
n_samples = 4 * sr
reference_length = 5

dataset_list = [line.rstrip('\n') for line in
                        open(os.path.abspath(os.path.expanduser("/home/quhongling/experiments/Network-test/dev_dataset_path.txt")), "r")]
writer = visualization.writer("/home/quhongling/experiments/Network-test/dataset_tmp/logs")

for item in tqdm(range(1260)):
    mixture_path, target_path, reference_path = dataset_list[item].split(" ")

    mixture = load_wav(mixture_path)
    target = load_wav(target_path)
    reference = load_wav(reference_path)

    print("\n", item + 1, "it\n", "mixture:", mixture, "\ntarget:", target, "\nreference:", reference, "\n")

    if len(reference) > (sr * reference_length):
        start = np.random.randint(len(reference) - sr * reference_length + 1)
        end = start + sr * reference_length
        reference = reference[start:end]
    else:
        reference = np.pad(reference, (0, sr * reference_length - len(reference)))

    mixture, target = sample_fixed_length_data_aligned(mixture, target, n_samples)

    print("after sample:\n", item+1, "it\n", "mixture:", mixture, "\ntarget:", target, "\nreference:", reference, "\n")

    writer.add_audio(f"Speech/{item + 1}_Mixture", mixture, 1, sample_rate=sr)
    writer.add_audio(f"Speech/{item + 1}_Target", target, 1, sample_rate=sr)
    writer.add_audio(f"Speech/{item + 1}_Reference", reference, 1, sample_rate=sr)
