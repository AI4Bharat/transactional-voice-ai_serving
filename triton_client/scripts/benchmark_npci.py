# python benchmark_npci.py --gt-file /home/npci/NPCI/transactional-voice-ai_serving/triton_client/data/npci_pipeline_benchmark/en-benchmark-v0109-fixed.csv --audio-folder /home/npci/NPCI/transactional-voice-ai_serving/triton_client/data/npci_pipeline_benchmark/audio --lang en --savefile /home/npci/NPCI/transactional-voice-ai_serving/triton_client/data/npci_pipeline_benchmark/results/temp.csv --batchsize 16

import os
import numpy as np
import soundfile as sf
import librosa
import math
import tritonclient.grpc as grpc_client
import sys
import time
import argparse
import pandas as pd
import json
import requests
from tqdm import tqdm
from sklearn.metrics import classification_report
from collections import defaultdict

greedy=False
dynamic_hotwords=False

def calc_entity_metrics(true_entities, pred_entities):
    n_TP = defaultdict(lambda: 0)
    n_FP = defaultdict(lambda: 0)
    n_FN = defaultdict(lambda: 0)
    n_count = defaultdict(lambda: 0)
    count = len(true_entities)
    is_correct = list()
    for i in range(count):
        pred = pred_entities[i]
        true = true_entities[i]
        is_correct.append(true == pred)
        for p in pred:
            if p in true:
                n_TP[p.split("-")[0]] += 1
            else:
                n_FP[p.split("-")[0].strip()] += 1
        for t in true:
            n_count[t.split("-")[0].strip()] += 1
            if t not in pred:
                n_FN[t.split("-")[0]] += 1

    entity_types = sorted(
        list(set(list(n_TP.keys()) + list(n_FP.keys()) + list(n_FN.keys())))
    )
    entity_report = list()
    for ent in entity_types:
        try:
            precision = (n_TP[ent]) / (n_TP[ent] + n_FP[ent])
        except ZeroDivisionError:
            precision = 0
        try:
            recall = (n_TP[ent]) / (n_TP[ent] + n_FN[ent])
        except ZeroDivisionError:
            recall = 0
        try:
            f1 = 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            f1 = 0
        precision, recall, f1 = (
            round(precision * 100),
            round(recall * 100),
            round(f1 * 100),
        )
        entity_report.append([ent, n_count[ent], precision, recall, f1])
    entity_report.append(
        ["Total", count, "", "", round(100 * sum(is_correct) / len(is_correct))]
    )
    entity_report_df = pd.DataFrame(
        entity_report,
        columns=["Entity Type", "Count", "Precision", "Recall", "F1 Score"],
    )
    return entity_report_df

def _convert_samples_to_float32(samples):
    """Convert sample type to float32.
    Audio sample type is usually integer or float-point.
    Integers will be scaled to [-1, 1] in float32.
    """
    float32_samples = samples.astype('float32')
    if samples.dtype in np.sctypes['int']:
        bits = np.iinfo(samples.dtype).bits
        float32_samples *= 1.0 / 2 ** (bits - 1)
    elif samples.dtype in np.sctypes['float']:
        pass
    else:
        raise TypeError("Unsupported sample type: %s." % samples.dtype)
    return float32_samples

def load_wav(path):
    if False: # Not required, above method works fine
        with sf.SoundFile(path, 'r') as f:
            dtype = 'float32'
            sample_rate = f.samplerate
            samples = f.read(dtype=dtype)
        samples = _convert_samples_to_float32(samples)
        samples = samples.transpose()
        samples = librosa.core.resample(samples, orig_sr=sample_rate, target_sr=16000)
        samples = samples.transpose()
        return samples
    else:
        audio, _ = librosa.load(path, sr=16000)
        return audio

def batchify(arr, batch_size=1):
    num_batches = math.ceil(len(arr) / batch_size)
    return [arr[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]

def pad_batch(batch_data):
    batch_data_lens = np.asarray([len(data) for data in batch_data], dtype=np.int32)
    max_length = max(batch_data_lens)
    batch_size = len(batch_data)

    padded_zero_array = np.zeros((batch_size,max_length),dtype=np.float32)

    for idx, data in enumerate(batch_data):
        padded_zero_array[idx,0:batch_data_lens[idx]] = data

    return padded_zero_array, np.reshape(batch_data_lens,[-1,1])


def download_audio(URL, audio_folder):
    audio_name = URL.rsplit("/")[-1][:-4]
    save_path = os.path.join(audio_folder, "{}.wav".format(audio_name))
    try:
        audio = requests.get(URL).content
        with open(save_path, "wb") as f:
            f.write(audio)
    except:
        return
def normalize_entities(ent_list):
    normalized_ent = list()
    for ent in ent_list:
        ent_type, ent_val = ent.split("-", 1)
        ent_type = ent_type.strip().lower()
        ent_val = ent_val.strip().lower().replace(" ", "")
        normalized_ent.append("{}-{}".format(ent_type, ent_val))
    return normalized_ent

def retain(true_ent_list):
    """
    Retain a sample if it is filled
    If it is not filled and contains nan or "enter xyz", drop
    """
    for ent in true_ent_list:
        if "nan" in true_ent_list:
            return False
        if "enter" in true_ent_list:
            return False
        if "select" in true_ent_list:
            return False
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", required=True)
    parser.add_argument("--audio-folder", required=True)
    parser.add_argument("--batchsize", default=1, type=int)
    parser.add_argument("--lang", required=True)
    parser.add_argument("--savefile", required=True)
    args = parser.parse_args()

    # df = pd.read_csv(args.gt_file, nrows=50)
    df = pd.read_csv(args.gt_file)
    df = df.dropna()
    df = df.reset_index(drop=True)
    df = df[df["True Entities"].apply(retain)]
    df = df.reset_index(drop=True)
    print("Number of samples:", len(df))
    df["audios"] = df["URL"].apply(lambda x: os.path.join(args.audio_folder, x.split("/")[-1]))
    df["True Entities"] = df["True Entities"].apply(json.loads)

    # Download audio files
    print("Checking audio files")
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if os.path.isfile(row["audios"]):
            continue
        download_audio(row["URL"])
    df = df[df["audios"].apply(os.path.isfile)]
    ##########################################
    # df = df.sample(frac=1)
    # df = df.iloc[:10]
    # df = df.reset_index(drop=True)
    ##########################################
    print("Number of samples with audios:", len(df))
    print()
    # -----------------  No sorting -----------------
    raw_audio_data = [load_wav(audio) for audio in df["audios"]]
    batches = batchify(raw_audio_data, batch_size=args.batchsize)
    
    # -----------------  With sorting TODO: bug with sorting algorithm  -----------------
    # raw_audio_data = []
    # audio_lens = []
    # for audio in df["audios"]:
    #     np_audio = load_wav(audio)
    #     raw_audio_data.append(np_audio)
    #     audio_lens.append(len(np_audio))
    # df["audio_lens"] = audio_lens
    # df.sort_values("audio_lens", inplace=True)
    # batches = batchify(sorted(raw_audio_data, key=len), batch_size=args.batchsize)

    triton_grpc_client = grpc_client.InferenceServerClient(url='localhost:8001', verbose=False)

    results_intent = list()
    results_entities = list()
    results_transcript = list()
    results_transcript_itn = list()
    inference_time = 0
    for i in tqdm(range(len(batches)), total=len(batches)):
        if args.batchsize == 1:
            audio_signal = np.array(batches[i])
            audio_len = np.asarray([[len(audio_signal[0])]], dtype=np.int32)
            # print(audio_signal.shape, audio_len.shape)
        else:
            audio_signal, audio_len = pad_batch(batches[i])            
        
        input0 = grpc_client.InferInput("AUDIO_SIGNAL", audio_signal.shape, "FP32")
        input0.set_data_from_numpy(audio_signal)
        input1 = grpc_client.InferInput("NUM_SAMPLES", audio_len.shape, "INT32")
        input1.set_data_from_numpy(audio_len.astype('int32'))
        output0 = grpc_client.InferRequestedOutput('TRANSCRIPTS_ASR')
        output1 = grpc_client.InferRequestedOutput('TRANSCRIPTS_ITN')
        output2 = grpc_client.InferRequestedOutput('LABELS_INTENT')
        output3 = grpc_client.InferRequestedOutput('JSON_ENTITY')
        outputs=[output0, output1, output2, output3]
        
        if greedy:
            inputs = [input0, input1]
            response = triton_grpc_client.infer("pipeline_greedy_ensemble_EN", model_version='1',inputs=inputs, request_id=str(1), outputs=outputs)
        else:
            if dynamic_hotwords:
                hotword_list=["rupees"]
                input2 = grpc_client.InferInput("HOTWORD_LIST", [len(audio_len),len(hotword_list)], "BYTES")
                input2.set_data_from_numpy(np.array([hotword_list*len(audio_len)]).astype('object').reshape([len(audio_len),-1]))
                input3 = grpc_client.InferInput("HOTWORD_WEIGHT", [len(audio_len),1], "FP32")
                input3.set_data_from_numpy(np.array([[10.]*len(audio_len)], dtype=np.float32).reshape([len(audio_len),-1]))
                inputs = [input0, input1, input2, input3]
            else:
                inputs = [input0, input1]
            response = triton_grpc_client.infer("pipeline_pyctc_ensemble_EN", model_version='1',inputs=inputs, request_id=str(1), outputs=outputs)

        result_response = response.get_response()
        batch_result_asr = response.as_numpy("TRANSCRIPTS_ASR")
        batch_result_asr_itn = response.as_numpy("TRANSCRIPTS_ITN")
        batch_result_intent = response.as_numpy('LABELS_INTENT')
        batch_result_entity = response.as_numpy('JSON_ENTITY')
        
        for i,sample_result in enumerate(batch_result_asr):
            results_transcript.append(sample_result[0].decode("utf-8"))
            results_transcript_itn.append(batch_result_asr_itn[i][0].decode("utf-8"))
            results_intent.append(batch_result_intent[i][0].decode("utf-8"))
            ner_result = json.loads(batch_result_entity[i][0].decode("utf-8"))
            results_entities.append(ner_result)

    print(f"Time to infer {len(df)} samples: {inference_time} s")

    df["Transcript"] = results_transcript
    df["Transcript ITN"] = results_transcript_itn
    df["Pred Intent"] = results_intent
    df["Pred Entities"] = results_entities
    df["Pred Entities"] = df["Pred Entities"].apply(lambda entities: [f"{ent['entity']}-{ent['value']}" for ent in entities])
    
    df.drop(columns="audios")
    df.to_csv(args.savefile, index=False)

    print("\n=================================Intent Report=================================\n")
    print(classification_report(df["True Intent"].apply(str), df["Pred Intent"].apply(str)))
    true_entities = df["True Entities"].apply(normalize_entities)
    pred_entities = (
        df["Pred Entities"].apply(normalize_entities)
    )
    print("\n=================================Entity Report=================================\n")
    print(calc_entity_metrics(true_entities, pred_entities))
