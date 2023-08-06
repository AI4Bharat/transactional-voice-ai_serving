# python benchmark_npci_concurrent.py --gt-file ../data/ta/npci_pipeline_benchmark/ta-benchmark-07-19.csv --audio-folder ../data/ta/npci_pipeline_benchmark/audio --savefile ../data/ta/npci_pipeline_benchmark/results/temp.csv --lang "ta" --batchsize 1
# python benchmark_npci_concurrent.py --gt-file ../data/hi/npci_pipeline_benchmark/hi-benchmark-v0109-fixed.csv --audio-folder ../data/hi/npci_pipeline_benchmark/audio --savefile ../data/hi/npci_pipeline_benchmark/results/temp.csv --lang "hi" --batchsize 1
# python benchmark_npci_concurrent.py --gt-file ../data/en/npci_pipeline_benchmark/en-benchmark-v0109-fixed.csv --audio-folder ../data/en/npci_pipeline_benchmark/audio --savefile ../data/en/npci_pipeline_benchmark/results/temp.csv --lang "en" --batchsize 1
from concurrent.futures import ThreadPoolExecutor
import os
import argparse
import pandas as pd
import json
import requests
from tqdm import tqdm
from sklearn.metrics import classification_report
from collections import defaultdict
import base64

greedy=False
dynamic_hotwords=False
global lang
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

def load_wav(path):
    with open(path, "rb") as f:
        audio_content: str = base64.b64encode(f.read()).decode("utf-8")
    return audio_content
      
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

def send_request(audio):
    inference_cfg: dict = {
        "language": {
            "sourceLanguage": lang
        },
        "audioFormat": "wav",
        "postProcessors": ["tag_entities"]
    }

    inference_inputs: dict = [
        {
            "audioContent": audio
        }
    ]
    inference_url = "http://localhost:8008/api"
    
    # control_config = {
    #     "dataTracking": False,
    # }
    http_headers: dict = {
        "authorization": "sample_secret_key_for_demo",
    }
    response: requests.Response = requests.post(
        url=inference_url,
        headers=http_headers,
        json={
            "config": inference_cfg,
            "audio": inference_inputs,
            # "controlConfig": control_config,
        }
    )

    if response.status_code != 200:
        print(f"Request failed with response.text: {response.text[:500]} and status_code: {response.status_code}")
        return {}
    
    return response.json()["output"][0]
    

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
    os.makedirs(args.audio_folder, exist_ok=True)
    os.makedirs(os.path.dirname(args.savefile), exist_ok=True)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if os.path.isfile(row["audios"]):
            continue
        download_audio(row["URL"], args.audio_folder)
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
    total_audio_dur = sum(map(lambda x:len(x)/16000, raw_audio_data))
    print(f"Total audio duration (in hrs) - {total_audio_dur/3600:.2f} | avg. duration (in s) - {total_audio_dur/len(raw_audio_data):.2f}")
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
    
    results_transcript = list()
    results_transcript_itn = list()
    results_intent = list()
    results_entities = list()
    inference_time = 0
    lang = args.lang
    with ThreadPoolExecutor(max_workers=8) as executor:
        result_responses = list(tqdm(executor.map(send_request, raw_audio_data), total=len(raw_audio_data)))
    
    for response in result_responses:
        results_transcript.append(response["source"])
        results_intent.append(response["intent"])
        results_entities.append(response["entities"])

    print(f"Time to infer {len(df)} samples: {inference_time} s")

    df["Transcript"] = results_transcript
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
