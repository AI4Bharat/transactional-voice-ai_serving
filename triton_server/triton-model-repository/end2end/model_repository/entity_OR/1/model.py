import triton_python_backend_utils as pb_utils
from entity_utils import EntityRecognizer
import json
import numpy as np
import os

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
        base_path = os.path.join(args["model_repository"], args["model_version"])
        self.entity_recognizer = EntityRecognizer("or", base_path)
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "entities")
        
        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

    def execute(self, requests):
        responses = []
        for request in requests:
            query = pb_utils.get_input_tensor_by_name(request, "input_text").as_numpy()
            query_itn = pb_utils.get_input_tensor_by_name(request, "input_text_itn")
            if query_itn is not None:
                itn_s = query_itn.as_numpy()
            sent_itn = None
            entities_list = []
            for i,s in enumerate(query):
                sent = s[0].decode("utf-8")
                if query_itn is not None:
                    sent_itn = itn_s[i][0].decode("utf-8")

                entities = self.entity_recognizer.predict(
                        sent, sent_itn
                    )
                entities_list.append(json.dumps(entities))
                
            out_numpy = np.array(entities_list).astype(self.output0_dtype)

            out_tensor_0 = pb_utils.Tensor("entities", out_numpy)
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)
        return responses
