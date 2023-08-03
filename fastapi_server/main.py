import os
import io
import json
import base64
import datetime
import shortuuid
from urllib.request import urlopen

from fastapi import FastAPI, Depends, Header, HTTPException, Request, Response, status
from fastapi.security import APIKeyHeader, HTTPBearer
from fastapi.security.http import HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from schema import *
from utils import batchify, get_raw_audio_from_file_bytes

# ## Read environment variables only during development purpose ##
# from dotenv import load_dotenv
# load_dotenv()

STANDARD_SAMPLING_RATE = int(os.environ["STANDARD_SAMPLING_RATE"])
STANDARD_BATCH_SIZE = int(os.environ["STANDARD_BATCH_SIZE"])
INFERENCE_SERVER_HOST = os.environ["INFERENCE_SERVER_HOST"]
DEFAULT_API_KEY_VALUE = os.environ["DEFAULT_API_KEY_VALUE"]

# Logging setup
ENABLE_LOGGING = os.environ.get("ENABLE_LOGGING", "false").lower() == "true"
if ENABLE_LOGGING:
    from azure.storage.blob import BlobServiceClient

    azure_account_url = f'https://{os.environ["AZURE_BLOB_STORE_NAME"]}.blob.core.windows.net'
    blob_service_client = BlobServiceClient(
        azure_account_url, credential=os.environ["AZURE_STORAGE_ACCESS_KEY"]
    )

    AZURE_BLOB_CONTAINER = os.environ["AZURE_BLOB_CONTAINER"]

## Initialize Triton client for a worker ##
from inference_client import InferenceClient
inference_client = InferenceClient(INFERENCE_SERVER_HOST)

## Create FastAPI app ##

def AuthProvider(
    request: Request,
    credentials_key: str = Depends(APIKeyHeader(name="Authorization")),
):
    validate_status = credentials_key and credentials_key == DEFAULT_API_KEY_VALUE
    if not validate_status:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"message": "Not authenticated"},
        )


api = FastAPI(
    title="NPCI ASR Inference API",
    description="Backend API for communicating with ASR models",
    dependencies=[
        Depends(AuthProvider),
    ],
)
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    # allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

## API Endpoints ##
@api.post("/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest, response: Response):
    language = request.config.language.sourceLanguage
    enable_logging = ENABLE_LOGGING and request.controlConfig.dataTracking
    raw_audio_list, metadata_list = [], []

    for input_index, input_item in enumerate(request.audio):
        
        if input_item.audioContent:
            file_bytes = base64.b64decode(input_item.audioContent)
        elif input_item.audioUri:
            file_bytes = urlopen(input_item.audioUri).read()
        else:
            response.status_code = status.HTTP_400_BAD_REQUEST
            return InferenceResponse(
                status=ResponseStatus(
                    success=False,
                    message=f"Neither `audioContent` nor `audioUri` found in `audio` input_index: {input_index}",
                ),
            )
        
        current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d/%H:%M:%S")
        metadata = {
            "timestamp": current_timestamp,
            "input_id": f"{current_timestamp}/{shortuuid.uuid()}",
            "language": language,
        }

        if enable_logging:
            audio_blob_path = f"{metadata['input_id']}/audio.{request.config.audioFormat}"
            blob_client = blob_service_client.get_blob_client(container=AZURE_BLOB_CONTAINER, blob=audio_blob_path)
            blob_client.upload_blob(file_bytes)

        raw_audio = get_raw_audio_from_file_bytes(file_bytes, standard_sampling_rate=STANDARD_SAMPLING_RATE)

        # For now, audio is small in size from NPCI, hence no VAD is required. So proceed directly without splitting
        raw_audio_list.append(raw_audio)
        metadata_list.append(metadata)
    
    final_results = []
    batches = batchify(raw_audio_list, batch_size=STANDARD_BATCH_SIZE)
    for i in range(len(batches)):
        # TODO: Add try-catch and handle errors, as well as log it
        batch_result = inference_client.run_batch_inference(batch=batches[i], lang_code=language, batch_size=STANDARD_BATCH_SIZE)
        
        for item_index, result_json in enumerate(batch_result):
            input_index = i*STANDARD_BATCH_SIZE + item_index

            # Convert intermediate format to final format
            result = InferenceResult(
                input_id=metadata_list[input_index]["input_id"],
                transcript=Transcript(raw=result_json["transcript"], itn=result_json["transcript_itn"]),
                intent=Intent(recommended_tag=result_json["intent"], original_tag=result_json["intent_orig"], probability=float(result_json["intent_prob"])),
                entities=[
                    Entity(
                        tag=entity["entity"],
                        substring=entity["word"],
                        start_index=entity["start"],
                        end_index=entity["end"],
                        extracted_value=entity["value"]
                    ) for entity in result_json["entities"]
                ]
            )
            final_results.append(result)

            if enable_logging:
                metadata_list[input_index]["result"] = result.model_dump(mode="json")
                del metadata_list[input_index]["result"]["input_id"] # Remove redundant field
                metadata_blob_path = f"{metadata['input_id']}/metadata.json"
                blob_client = blob_service_client.get_blob_client(container=AZURE_BLOB_CONTAINER, blob=metadata_blob_path)
                blob_client.upload_blob(json.dumps(metadata_list[input_index], indent=4))
    
    return InferenceResponse(
        output=final_results,
        status=ResponseStatus(success=True),
    )
