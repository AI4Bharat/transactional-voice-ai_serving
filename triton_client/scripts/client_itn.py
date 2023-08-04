import os
import numpy as np
from glob import glob
import sys
import tritonclient.http as http_client
# import triton_python_backend_utils as pb_utils

import time

triton_client = http_client.InferenceServerClient(
    url="localhost:8000",
)
headers = {}
headers["Authorization"] = f"Bearer GougU6OLYC64cAJH4wbjSMOUbh6cidmg"

# Check status of triton server
health_ctx = triton_client.is_server_ready(headers=headers)
print("Is server ready - {}".format(health_ctx))

sent_list = ["transfer हंड्रेड rupees", "recharge electricity"]
# sent_list = ["transfer hundred rupees", "recharge electricity"]
start_time = time.time()
input0 = http_client.InferInput("INPUT_TEXT", [2, 1], "BYTES")
input0_data = np.array([sent.encode('utf-8') for sent in sent_list], dtype=np.object_).reshape([2,1])
input0.set_data_from_numpy(input0_data)

output0 = http_client.InferRequestedOutput('OUTPUT_TEXT')
response = triton_client.infer("itn_HI", model_version='1', inputs=[input0],\
    request_id=str(1), outputs=[output0])
result_response = response.get_response()
encoded_result = response.as_numpy('OUTPUT_TEXT')
print("Total Time Taken {}".format(time.time() - start_time))
for result in encoded_result:
    print(result[0].decode("utf-8"))
