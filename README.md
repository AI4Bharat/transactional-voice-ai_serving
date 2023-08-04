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
TBD
