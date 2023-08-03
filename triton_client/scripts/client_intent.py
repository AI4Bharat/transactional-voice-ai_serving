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

import time

# triton_grpc_client = http_client.InferenceServerClient(url='localhost:8881', verbose=False)
triton_client = http_client.InferenceServerClient(
    # url="hi-asr--whisper-medium-t4.centralindia.inference.ml.azure.com",
    url="localhost:8000",
    # ssl=True,
    # ssl_context_factory=gevent.ssl._create_default_https_context,
    # connection_timeout=180.0,
    # network_timeout=180.0
)
headers = {}
headers["Authorization"] = f"Bearer GougU6OLYC64cAJH4wbjSMOUbh6cidmg"

# Check status of triton server
health_ctx = triton_client.is_server_ready(headers=headers)
print("Is server ready - {}".format(health_ctx))

sent_list = ["transfer 100 rupees", "recharge electricity"]
start_time = time.time()
input0 = http_client.InferInput("input_text", [2, 1], "BYTES")
input0_data = np.array([sent.encode('utf-8') for sent in sent_list], dtype=np.object_).reshape([2,1])
input0.set_data_from_numpy(input0_data)

output0 = http_client.InferRequestedOutput('labels')
response = triton_client.infer("intent_ensemble", model_version='1', inputs=[input0],\
    request_id=str(1), outputs=[output0])
result_response = response.get_response()
encoded_result = response.as_numpy('labels')
print("Total Time Taken {}".format(time.time() - start_time))
for result in encoded_result:
    print(result[0].decode("utf-8"))
