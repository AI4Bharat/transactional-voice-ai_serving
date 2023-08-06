# python single_file_inference.py "ta"

import base64
import requests
import sys

lang = sys.argv[1]

class Client:
    def __init__(self, api_key: str, consent_for_data_tracking: bool = True, inference_url: str = "http://localhost:8008/api") -> None:
        self.http_headers: dict = {
            "authorization": api_key,
        }
        self.inference_url = inference_url
        # self.control_config = {
        #     "dataTracking": consent_for_data_tracking,
        # }
    
    def run_inference(self, audio_path: str, src_lang_code: str) -> dict:
        # Read audio from file and encode it as string so that it can transmitted in a JSON payload
        with open(audio_path, "rb") as f:
            audio_content: str = base64.b64encode(f.read()).decode("utf-8")
        
        inference_cfg: dict = {
            "language": {
                "sourceLanguage": src_lang_code
            },
            "audioFormat": "wav",
            "postProcessors": ["tag_entities"]
        }

        inference_inputs: dict = [
            {
                "audioContent": audio_content
            }
        ]

        response: requests.Response = requests.post(
            url=self.inference_url,
            headers=self.http_headers,
            json={
                "config": inference_cfg,
                "audio": inference_inputs,
                # "controlConfig": self.control_config,
            }
        )

        if response.status_code != 200:
            print(f"Request failed with response.text: {response.text[:500]} and status_code: {response.status_code}")
            return {}

        return response.json()["output"][0]

if __name__ == "__main__":
    API_KEY = "sample_secret_key_for_demo"
    FILENAME = f"../data/{lang}/{lang}.wav"
    LANG_CODE = f"{lang}"
    ALLOW_LOGGING_ON_SERVER = True

    client = Client(api_key=API_KEY, consent_for_data_tracking=ALLOW_LOGGING_ON_SERVER, inference_url="http://localhost:8008/api")
    result = client.run_inference(audio_path=FILENAME, src_lang_code=LANG_CODE)
    print(result)
