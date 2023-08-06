from inverse_text_normalization.ta.run_predict import inverse_normalize_text as ta_itn
import triton_python_backend_utils as pb_utils
import numpy as np
import json

def format_numbers_with_commas(sent, lang):
    words = []
    for word in sent.split(' '):
        word_contains_digit = any(map(str.isdigit, word))
        currency_sign = ''
        if word_contains_digit:
            if len(word) > 4 and ':' not in word:
                pos_of_first_digit_in_word = list(map(str.isdigit, word)).index(True)

                if pos_of_first_digit_in_word != 0:  # word can be like $90,00,936.59
                    currency_sign = word[:pos_of_first_digit_in_word]
                    word = word[pos_of_first_digit_in_word:]

                s, *d = str(word).partition(".")
                # getting [num_before_decimal_point, decimal_point, num_after_decimal_point]
                if lang == 'hi':
                    # adding commas after every 2 digits after the last 3 digits
                    r = "".join([s[x - 2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
                else:
                    r = "".join([s[x - 3:x] for x in range(-3, -len(s), -3)][::-1] + [s[-3:]])

                word = "".join([r] + d)  # joining decimal points as is

                if currency_sign:
                    word = currency_sign + word
                words.append(word)
            else:
                words.append(word)
        else:
            words.append(word)
    return ' '.join(words)


def inverse_normalize_text(text_list):
    try:
        itn_results = ta_itn(text_list)
        text_list = [format_numbers_with_commas(sent=sent, lang='ta') for sent in itn_results]
    except Exception as e:
        print(e)
    return text_list
    
class TritonPythonModel:

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
        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args['model_config'])
        
        # Get OUTPUT0 configuration
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "OUTPUT_TEXT")
        
        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

    def execute(self, requests):
        """
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []
        for request in requests:
            # B X T
            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT_TEXT")
            input_text = [s[0].decode() for s in in_0.as_numpy()]
            # B X 1
            sent = np.array(inverse_normalize_text(input_text))
            
            out_tensor_0 = pb_utils.Tensor("OUTPUT_TEXT", sent.astype(self.output0_dtype))
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)
        return responses
