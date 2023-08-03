import math
import io
import soundfile as sf
import numpy as np
import scipy.signal as sps

def batchify(arr, batch_size=1):
    num_batches = math.ceil(len(arr) / batch_size)
    return [arr[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]

def pad_batch(batch_data):
    batch_data_lens = np.asarray([len(data) for data in batch_data], dtype=np.int32)
    max_length = max(batch_data_lens)
    batch_size = len(batch_data)

    padded_zero_array = np.zeros((batch_size, max_length), dtype=np.float32)

    for idx, data in enumerate(batch_data):
        padded_zero_array[idx, 0:batch_data_lens[idx]] = data

    return padded_zero_array, np.reshape(batch_data_lens, [-1,1])

def get_raw_audio_from_file_bytes(file_bytes, standard_sampling_rate):
    file_handle = io.BytesIO(file_bytes)
    data, sampling_rate = sf.read(file_handle)
    data = data.tolist()
    raw_audio = np.array(data)  # in float64

    if len(raw_audio.shape) > 1:  # Stereo to mono
        raw_audio = raw_audio.sum(axis=1) / 2

    if sampling_rate != standard_sampling_rate:
        number_of_samples = round(
            len(raw_audio) * float(standard_sampling_rate) / sampling_rate
        )
        raw_audio = sps.resample(raw_audio, number_of_samples).astype("float32")
    
    return raw_audio
