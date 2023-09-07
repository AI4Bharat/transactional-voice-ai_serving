<h1 align="center">Transactional Voice AI <i>(Serving)</i> </h1>

<p align="center"><u>Modular</u>, <u>Scalable</u> and  <u>Optimized</u> deployment code for all the Transactional Voice AI modules namely Automatic Speech Recognition (ASR), Inverse Text Normalization (ITN), Intent and Entity Recognition using <a href="https://github.com/triton-inference-server/server">Triton Inference Server</a> and <a href="https://github.com/tiangolo/fastapi">FastAPI</a>.</p>
<p align="center">
  For more info on Transactional Voice AI development codebase, refer <a href="https://github.com/AI4Bharat/transactional-voice-ai">here</a>.
  <br> <br>
  <a href="#prerequisites">Prerequisites</a> •
  <a href="#setup">Setup</a> •
  <a href="#schema">Schema</a> •
  <a href="#test">Test</a> •
  <a href="#benchmark">Benchmark</a>
</p>

## Prerequisites

> Make sure that the system have NVIDIA GPU card(s) and corresponding drivers installed.

- Docker (and Docker Compose) - Follow steps outlined [here](https://docs.docker.com/engine/install/).
- Install `nvidia-container-toolkit` - [Official Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#docker) *(Note: it also contains steps on how to install Docker too!)*

## Setup

> Clone the repository: 
```
git clone https://github.com/AI4Bharat/transactional-voice-ai_serving.git
cd transactional-voice-ai_serving
```
> Copy (and modify, if required) `.env` file:
```
cp .env.example .env
```
> Build and Run Docker Containers using  `docker compose`:
```
docker compose up --build
```
_This will start two servers, one **Triton Inference Server** and its wrapper **FastAPI server** which also acts as an entrypoint to every request._ 

## Schema
### Payload
The payload to the server must follow the following schema:
```json
{
  "config": {
    "language": {
      "sourceLanguage": "en"
    },
    "transcriptionFormat": {
      "value": "transcript"
    },
    "audioFormat": "wav",
    "samplingRate": 8000,
    "postProcessors": [
      "tag_entities"
    ]
  },
  "audio": [
    {
      "audioUri": "https://t3638486.p.clickup-attachments.com/t3638486/b6f63475-a96f-4c25-be45-0495946d440e/8797501890_mobile_number440_08_09_2022_20_46_25.wav"
    }
  ]
}
```
The ```audioUri``` should contain a link to a wav file. Instead of an URI, one can also provide audio in base64 format by using ```audioContent``` key:
```json
{
  "config": {
    "language": {
      "sourceLanguage": "en"
    },
    "transcriptionFormat": {
      "value": "transcript"
    },
    "audioFormat": "wav",
    "samplingRate": 8000,
    "postProcessors": [
      "tag_entities"
    ]
  },
  "audio": [
    {
      "audioContent": "GkXfo59ChoEBQveBAULygQRC84EIQoKEd2VibUKHgQRChYECGFOAZw…"
    }
  ]
}
```
### Response
The server responds with the ASR transcript along with intent and entity predictions. The intent is provided as a single string indicating the intent name, while entity is a list containing one dictionary per predicted entity giving the information regarding entity type, word, value, start and end indices.
```json
{
  "status": "SUCCESS",
  "output": [
    {
      "source": "please transfer 200 rupees to mobile number 9998887776 from my sbi account",
      "entities": [
        {
          "entity": "bank_name",
          "word": "sbi account",
          "start": 42,
          "end": 45,
          "value": "state_bank"
        },
        {
          "entity": "amount_of_money",
          "word": "200 rupees",
          "start": 17,
          "end": 27,
          "value": "200"
        },
        {
          "entity": "mobile_number",
          "word": "9998887776",
          "start": 38,
          "end": 48,
          "value": "9998887776"
        }
      ],
      "id": "abcdefghij1234567890AB",
      "intent": "p2p_transfer"
    }
  ]
}
```

## Test

> Test the complete pipeline using the python client on a sample `.wav` file: 
```
cd fastapi_client/scripts
python single_file_inference.py
```

## Benchmark

All the results shown below can be reproduced by running the following commands -
```
cd fastapi_client/scripts
```
> For Tamil language -
```
python benchmark_npci_concurrent.py --gt-file ../data/ta/npci_pipeline_benchmark/ta-benchmark-07-19.csv --audio-folder ../data/ta/npci_pipeline_benchmark/audio --savefile ../data/ta/npci_pipeline_benchmark/results/temp.csv --lang "ta" --batchsize 1
```
> For Hindi language -
```
python benchmark_npci_concurrent.py --gt-file ../data/hi/npci_pipeline_benchmark/hi-benchmark-v0109-fixed.csv --audio-folder ../data/hi/npci_pipeline_benchmark/audio --savefile ../data/hi/npci_pipeline_benchmark/results/temp.csv --lang "hi" --batchsize 1
```
> For English language -
```
python benchmark_npci_concurrent.py --gt-file ../data/en/npci_pipeline_benchmark/en-benchmark-v0109-fixed.csv --audio-folder ../data/en/npci_pipeline_benchmark/audio --savefile ../data/en/npci_pipeline_benchmark/results/temp.csv --lang "en" --batchsize 1
```

### Benchmark Data Statistics -

> *Data Source* - NPCI's collection of samples through IVRS 

> *Data Type* - **8Khz** single channel audio data, human-annotated entity/intent labels

| Language | # Utterances | Total Duration (hrs) | Average length (sec) | # Entities | # Intents |
| --- | --- | --- | --- | --- | --- |
| en | 2584 | 9.50 | 13.23 | 2239 | 894 |
| hi | 4411 | 8.9 | 12.88 | 2671 | 1279 |
| ta | 665 | 1.75 | 9.48 | 676 | - |

### Performance Statistics -
> *Metrics* - Accuracy for Intent Recognition and F1 Score for Entity Recognition

| Language | Intent Type | Intent Accuracy | Entity Type | Entity F1 Score |
| --- | --- | --- | --- | --- |
| en | p2p\_transfer | 85 | amount\_of\_money | 89 |
|    |    |    | bank\_name | 86 |
|    |    |    | mobile\_number | 88 |
| hi | p2p\_transfer | 87 | amount\_of\_money | 90 |
|    |    |    | bank\_name | 90 |
|    |    |    | mobile\_number | 86 |
| ta | p2p\_transfer | - | amount\_of\_money | 69 |
|    |    |    | bank\_name | 82 |
|    |    |    | mobile\_number | 75 |

### Runtime Statistics -

>*Setup1 (Default)* - **Single** instance of all the models loaded in GPU memory using Triton Inference Server, except for Pyctcdecode CPU module (8 instances)


| Lang | Hardware Type | Avg. GPU VRAM usage (GB) | Avg. GPU utilization | Avg. CPU RAM usage(GB) | Avg. CPU utilization | Total Time taken (s)/samples | Avg Latency (s) | Avg Throughput (RPS) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | **A100-80GB,** 16-core-110GB | 9.23 | 6.89 | 21.15 | 34.75 | 393/2584 | 1.19 | 5.92 |
| en | **T4-16GB,** 4-core-28GB | 4.16 | 16.47 | 23 | 83.95 | 759/2584 | 2.37 | 3.25 |
| hi | **A100-80GB,** 16-core-110GB | 11.8 | 5.19 | 22.26 | 13.48 | 968/4411 | 1.73 | 4.32 |
| hi | **T4-16GB,** 4-core-28GB | 5.2 | 16.6 | 24.28 | 51.58 | 1483/4411 | 2.72 | 2.94 |
| ta | **A100-80GB,** 16-core-110GB | 14.6 | 5.45 | 22.23 | 44.84 | 172/665 | 1.91 | 3.52 |
| ta | **T4-16GB,** 4-core-28GB | 6.15 | 11.4 | 24.49 | 93.22 | 381/665 | 4.77 | 1.63 |   

>*Setup2* - **Two** instances of all the models loaded in GPU memory using Triton Inference Server, except for Pyctcdecode CPU module (8 instances)

| Lang | Hardware Type | Avg. GPU VRAM usage (GB) | Avg. GPU utilization | Avg. CPU RAM usage(GB) | Avg. CPU utilization | Total Time taken (s)/samples | Avg Latency (s) | Avg Throughput (RPS) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| en | **A100-80GB,** 16-core-110GB | 13.3 | 11.35 | 25.04 | 51.86 | 273/2584 | 1.89 | 8.84 |
| hi | **A100-80GB,** 16-core-110GB | 17.1 | 10.94 | 26.7 | 25.6 | 968/4411 | 1.86 | 8.35 |
| ta | **A100-80GB,** 16-core-110GB | 13.3 | 6.3 | 24.73 | 53.91 | 147/665 | 3.22 | 4.09 |

> For higher throughput values, increase the count of `instance_group` in Triton's configuration or/and scale horizontally.

_Note: All the stats are for the end-to-end system including the FastAPI wrapper on top of Triton server._
