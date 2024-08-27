import os
import math
import json
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
import h5py

# Define constants
DATASET_PATH = Path(r"C:\Users\User\OneDrive - University of Canberra - STAFF\PhD\Studies\HRI\DataFinal")
HDF5_PATH = Path("data.h5")

SAMPLE_RATE = 22050
DURATION = 5  # seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(
    dataset_path,
    hdf5_path,
    n_mfcc=13,
    n_fft=2048,
    hop_length=512,
    num_segments=5,
):
    # Initialize lists to store data
    mfccs = []
    labels = []
    mappings = []

    # Iterate through all subdirectories in the dataset path
    for label_idx, situation_dir in enumerate(sorted(dataset_path.iterdir())):
        if situation_dir.is_dir():
            situation = situation_dir.name
            mappings.append(situation)
            print(f"\nProcessing situation: {situation}")

            # Process all audio files in the situation directory
            for audio_file in tqdm(list(situation_dir.glob('*.wav')) + list(situation_dir.glob('*.mp3'))):
                try:
                    signal, sr = librosa.load(audio_file, sr=SAMPLE_RATE)
                    track_duration = librosa.get_duration(y=signal, sr=sr)

                    if track_duration >= DURATION:
                        # Process segments
                        for segment in range(num_segments):
                            start_sample = int(segment * SAMPLES_PER_TRACK / num_segments)
                            end_sample = int(start_sample + SAMPLES_PER_TRACK / num_segments)

                            mfcc = librosa.feature.mfcc(
                                y=signal[start_sample:end_sample],
                                sr=sr,
                                n_mfcc=n_mfcc,
                                n_fft=n_fft,
                                hop_length=hop_length
                            )

                            if mfcc.shape[1] == math.ceil(SAMPLES_PER_TRACK / num_segments / hop_length):
                                mfccs.append(mfcc.T)
                                labels.append(label_idx)
                    else:
                        print(f"Skipping {audio_file.name}: audio is shorter than {DURATION} seconds.")
                except Exception as e:
                    print(f"Error processing {audio_file.name}: {e}")

    # Convert lists to numpy arrays
    mfccs = np.array(mfccs)
    labels = np.array(labels)

    # Save data to HDF5 file
    with h5py.File(hdf5_path, 'w') as hf:
        hf.create_dataset('mfcc', data=mfccs)
        hf.create_dataset('labels', data=labels)
        hf.create_dataset('mappings', data=np.array(mappings, dtype=h5py.string_dtype()))

    print(f"\nData successfully saved to {hdf5_path}")

if __name__ == "__main__":
    save_mfcc(
        dataset_path=DATASET_PATH,
        hdf5_path=HDF5_PATH,
        n_mfcc=40,  # Increased for better feature representation
        num_segments=10  # Increased segments for more samples
    )
