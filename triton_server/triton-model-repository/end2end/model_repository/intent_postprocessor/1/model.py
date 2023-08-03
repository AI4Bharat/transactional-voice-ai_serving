import numpy as np
from transformers import AutoTokenizer

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.


import triton_python_backend_utils as pb_utils



class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to intialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        model_name = 'ai4bharat/indic-bert'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        labels_to_ids = {'balance_check': 0, 'cancel': 1, 'confirm': 2, 'electricity_payment': 3, 'emi_collection_full': 4, 'emi_collection_partial': 5, 'fastag_recharge': 6, 'gas_payment': 7, 'inform': 8, 'insurance_renewal': 9, 'mobile_recharge_postpaid': 10, 'mobile_recharge_prepaid': 11, 'p2p_transfer': 12, 'petrol_payment': 13, 'upi_creation': 14}
        self.ids_to_labels = {
            intent_id: intent_label
            for intent_label, intent_id in labels_to_ids.items()
        }
        
    def execute(self, requests):
        """`execute` MUST be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference request is made
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        
        for request in requests:

            logits = pb_utils.get_input_tensor_by_name(request, 'logits').as_numpy().argmax(-1)

            batch_predictions = []
            
            for pred in logits:
                
                _prediction = self.ids_to_labels[pred]
                batch_predictions.append(_prediction)
            
            labels = np.array([l.encode('utf-8') for l in batch_predictions])
            labels_tensor = pb_utils.Tensor("labels", labels)
            
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[labels_tensor])
            responses.append(inference_response)
        
        return responses
    
