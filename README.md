<h1 align="center">Transcational Voice AI <i>(Serving)</i> </h1>

<p align="center"><u>Modular</u>, <u>Scalable</u> and  <u>Optimized</u> deployment code for all the Transactional Voice AI modules namely Automatic Speech Recognition (ASR), Inverse Text Normalization (ITN), Intent and Entity Recognition using <a href="https://github.com/triton-inference-server/server">Triton Inference Server</a> and <a href="https://github.com/tiangolo/fastapi">FastAPI</a>.</p>
<p align="center">
  For more info on Transactional Voice AI development codebase, refer <a href="https://ucr.docyard.ai/">here</a>.
  <br> <br>
  <a href="#prerequisites">Prerequisites</a> •
  <a href="#setup">Setup</a> •
  <a href="#test">Test</a> •
  <a href="#schema">Schema</a>
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

## Test
> Test the complete pipeline on a sample `.wav` file: 
```
cd fastapi_client/scripts
python single_file_inference.py
```

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
