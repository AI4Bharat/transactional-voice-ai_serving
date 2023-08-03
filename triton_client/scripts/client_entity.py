import os
import numpy as np
from glob import glob
import sys
# if sys.version_info.major == 3:
#     unicode = bytes
# import tritonclient.grpc as http_client
'''
# Added new lines
'''
import gevent.ssl
import tritonclient.http as http_client
# import triton_python_backend_utils as pb_utils
import json
import time

with_itn=True
triton_client = http_client.InferenceServerClient(
    url="localhost:8000",
)
headers = {}
headers["Authorization"] = f"Bearer GougU6OLYC64cAJH4wbjSMOUbh6cidmg"

# Check status of triton server
health_ctx = triton_client.is_server_ready(headers=headers)
print("Is server ready - {}".format(health_ctx))

sent_list = ["transfer hundred rupees", "recharge electricity"]
start_time = time.time()
input0 = http_client.InferInput("input_text", [2, 1], "BYTES")
input0_data = np.array([sent.encode('utf-8') for sent in sent_list], dtype=np.object_).reshape([2,1])
input0.set_data_from_numpy(input0_data)

if with_itn:
    itn_list = ["transfer 100 rupees", "recharge electricity"]
    input1 = http_client.InferInput("input_text_itn", [2, 1], "BYTES")
    input1_data = np.array([sent.encode('utf-8') for sent in itn_list], dtype=np.object_).reshape([2,1])
    input1.set_data_from_numpy(input1_data)
    inputs = [input0, input1]
else:
    inputs = [input0]

output0 = http_client.InferRequestedOutput('entities')
response = triton_client.infer("entity_EN", model_version='1', inputs=inputs,\
    request_id=str(1), outputs=[output0])
result_response = response.get_response()
encoded_result = response.as_numpy('entities')
print("Total Time Taken {}".format(time.time() - start_time))
for result in encoded_result:
    print(json.loads(result[0].decode("utf-8")))
