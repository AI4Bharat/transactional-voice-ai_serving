# python client_asr.py /home/npci/NPCI/transactional-voice-ai_serving/triton_client/data/npci_asr_benchmark/manifest.json 8
import pandas as pd
import json
import sys
import tritonclient.http as http_client
import time
import gevent.ssl
import math
from glob import glob
import numpy as np
import os
from tqdm import tqdm
from evaluate import load
import soundfile as sf
import librosa

file_name = sys.argv[1]
batch_size = int(sys.argv[2])
greedy=False
dynamic_hotwords=True

with open(file_name) as f:
    txt_file = f.readlines()[:100]
df = pd.DataFrame([json.loads(l.strip()) for l in txt_file])

def pad_batch(batch_data):
    batch_data_lens = np.asarray([len(data) for data in batch_data], dtype=np.int32)
    max_length = max(batch_data_lens)
    batch_size = len(batch_data)
    padded_zero_array = np.zeros((batch_size,max_length),dtype=np.float32)
    for idx, data in enumerate(batch_data):
        padded_zero_array[idx,0:batch_data_lens[idx]] = data

    return padded_zero_array, np.reshape(batch_data_lens,[-1,1])

def load_wav(path):
    # audio, _ = sf.read(path)
    audio, sr = librosa.load(path, sr=16000)
    return audio
def batchify(arr, batch_size=1):
    num_batches = math.ceil(len(arr) / batch_size)
    return [arr[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]

fnames = df["audio_filepath"].tolist()
# print(fnames)
raw_audio_data = [load_wav(fname) for fname in fnames]
#raw_audio_data = [load_wav(fname) for fname in fnames]
batches = batchify(raw_audio_data, batch_size=batch_size)

triton_http_client = http_client.InferenceServerClient(
    url="localhost:8000",
)

references = df["text"].tolist()
predictions = []
for i in tqdm(range(len(batches))):
    if batch_size == 1:
        audio_signal = np.array(batches[i])
        audio_len = np.asarray([[len(audio_signal[0])]], dtype=np.int32)
    else:
        audio_signal, audio_len = pad_batch(batches[i])

    input0 = http_client.InferInput("AUDIO_SIGNAL", audio_signal.shape, "FP32")
    input0.set_data_from_numpy(audio_signal)
    input1 = http_client.InferInput("NUM_SAMPLES", audio_len.shape, "INT32")
    input1.set_data_from_numpy(audio_len.astype('int32'))
    output0 = http_client.InferRequestedOutput('TRANSCRIPTS')
    
    
    if greedy:
        inputs = [input0, input1]
        response = triton_http_client.infer("asr_greedy_ensemble_HI", model_version='1',inputs=inputs, request_id=str(1), outputs=[output0],)
    else:
        if dynamic_hotwords:
            hotword_list=["rupees"]
            input2 = http_client.InferInput("HOTWORD_LIST", [len(audio_len),len(hotword_list)], "BYTES")
            input2.set_data_from_numpy(np.array([hotword_list*len(audio_len)]).astype('object').reshape([len(audio_len),-1]))
            input3 = http_client.InferInput("HOTWORD_WEIGHT", [len(audio_len),1], "FP32")
            input3.set_data_from_numpy(np.array([[10.]*len(audio_len)], dtype=np.float32).reshape([len(audio_len),-1]))
            inputs = [input0, input1, input2, input3]
        else:
            inputs = [input0, input1]
        response = triton_http_client.infer("asr_pyctc_ensemble_HI", model_version='1',inputs=inputs, request_id=str(1), outputs=[output0],)
    
    result_response = response.get_response()
    batch_result_asr = response.as_numpy("TRANSCRIPTS")
    for j,b in enumerate(batch_result_asr):
        print(references[i*batch_size+j], "||", b[0].decode("utf-8"))
        predictions.append(b[0].decode("utf-8"))

        
wer = load("wer")     
wer = wer.compute(predictions=predictions, references=references)
df[f"greedy_triton_{wer:.2f}"] = predictions
print(wer)

all_lines = [json.dumps(line)+"\n" for line in df.to_dict(orient="records")]
with open(file_name.replace(".json", "_greedy_triton.json"), 'w') as f:
    f.writelines(all_lines)
