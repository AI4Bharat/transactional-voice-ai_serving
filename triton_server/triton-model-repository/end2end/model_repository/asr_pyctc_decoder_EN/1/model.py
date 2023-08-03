from hotword_utils import hotword_to_fn
from pyctcdecode import build_ctcdecoder
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack
import json
import numpy as np
import os

class TritonPythonModel:
    def initialize(self, args):
        self.model_config = model_config = json.loads(args['model_config'])
        base_path = os.path.join(args["model_repository"], args["model_version"])
        self.vocab = ['<unk>', 's', 't', 'e', '▁the', 'd', '▁', '▁a', 'i', 'n', 'a', 'm', '▁to', 'y', 'o', 'ing', '▁and', 'er', 'p', 'u', '▁in', '▁of', "'", '▁i', '▁that', 'ed', 're', 'r', 'c', 'h', 'al', 'ar', 'f', '▁you', '▁s', '▁f', 'an', 'b', '▁it', 'l', 'w', 'is', '▁p', 'in', '▁we', '▁re', '▁be', 'es', 'g', 'or', '▁he', '▁c', 'ly', 'le', 'k', 'en', '▁for', '▁w', 'll', 'ur', 'ic', 'ri', '▁e', '▁so', 'on', 'ct', 've', '▁b', '▁g', '▁st', 'it', '▁t', '▁do', 'ra', '▁on', '▁was', '▁this', 'ent', 'th', 'ro', 'ce', '▁have', '▁de', '▁o', 'ter', '▁ma', '▁se', '▁co', '▁di', 'ation', '▁with', '▁not', '▁m', 'il', '▁me', 'us', 'ir', '▁are', 'v', '▁but', '▁pro', '▁th', 'ch', '▁con', 'ate', 'me', 'at', 'la', 'li', '▁they', 'ver', '▁go', '▁what', '▁ha', 'vi', '▁ne', '▁or', 'ive', '▁as', '▁there', '▁know', 'ment', 'un', 'lo', '▁su', '▁can', '▁is', '▁ex', '▁ch', '▁mo', 'ck', 'ul', '▁like', 'tion', 'el', '▁le', '▁one', 'ng', 'ci', '▁ca', '▁an', '▁all', 'ne', 'ge', '▁lo', 'x', 'ut', '▁la', '▁if', '▁at', '▁un', 'ol', 'qu', '▁no', '▁fa', 'as', '▁ho', 'ity', '▁just', '▁would', '▁about', '▁from', '▁ba', '▁v', 'mp', '▁think', '▁my', 'z', 'co', 'ad', '▁us', '▁will', '▁li', 'end', '▁by', 'ight', '▁some', '▁po', '▁his', 'ig', 'ry', '▁your', '▁our', '▁out', '▁pa', 'ff', '▁don', 'ru', '▁had', '▁te', '▁up', 'j', '▁when', '▁because', '▁which', '▁da', '▁get', 'age', '▁sp', '▁two', '▁bo', '▁say', 'sion', 'ction', '▁pre', '▁were', 'ence', '▁how', '▁time', '▁k', '▁who', '▁mi', '▁right', '▁comp', 'able', '▁she', '▁any', '▁more', 'ugh', '▁now', '▁other', '▁yeah', '▁app', 'ance', '▁uh', '▁also', '▁people', '▁part', '▁want', '▁very', 'ound', '▁work', '▁look', '▁comm', 'port', '▁year', '▁case', '▁court', '▁really', '▁said', 'side', '▁where', '▁could', '▁make', '▁even', '▁dr', '▁every', '▁those', '▁take', '▁ju', '▁three', '▁good', '▁first', '▁should', '▁point', 'q']
        self.static_hotwords_list = hotword_to_fn["entities-unique"](
                    lang="en", base_path=base_path)
        self.hotword_weight=10 # TODO: Move all the static variables to the config.pbtxt file as parameters
        self.lm_path=None
        self.alpha = 1
        self.beta = 1
        if self.lm_path is not None:
            self.decoder = build_ctcdecoder(self.vocab, self.lm_path, alpha=self.alpha, beta=self.beta)
        else:
            self.decoder = build_ctcdecoder(self.vocab)
        
        output0_config = pb_utils.get_output_config_by_name(
            model_config, "TRANSCRIPT")
        
        # Convert Triton types to numpy types
        self.output0_dtype = pb_utils.triton_string_to_numpy(
            output0_config['data_type'])

    def execute(self, requests):
        responses = []
        for request in requests:
            in_0 = pb_utils.get_input_tensor_by_name(request, "LOG_PROBS")
            in_1 = pb_utils.get_input_tensor_by_name(request, "HOTWORD_LIST")
            in_2 = pb_utils.get_input_tensor_by_name(request, "HOTWORD_WEIGHT")
            if in_1 is not None:
                hotword_list = in_1.as_numpy().tolist()
            
            if in_2 is not None:
                hotword_weight = from_dlpack(in_2.to_dlpack())
            
            logits = from_dlpack(in_0.to_dlpack())
            logits = logits.cpu().numpy()
            
            transcripts = []
            for i in range(len(logits)):
                # GET HOTWORDS LIST                
                hotword_l = self.static_hotwords_list 
                if in_1 is not None:
                    hotword_l += [hw.decode("UTF-8") for hw in hotword_list[i]]
                # GET HOTWORDS WEIGHT
                if in_2 is not None:                
                    hotword_w = hotword_weight[i].item()
                else:
                    hotword_w = self.hotword_weight

                transcript = self.decoder.decode(logits[i], hotwords=hotword_l, hotword_weight=hotword_w)
                transcripts.append(transcript.encode('utf-8'))
                # transcript = np.array([transcript])
            out_numpy = np.array(transcripts).astype(self.output0_dtype)

            out_tensor_0 = pb_utils.Tensor("TRANSCRIPT", out_numpy)
            inference_response = pb_utils.InferenceResponse(output_tensors=[out_tensor_0])
            responses.append(inference_response)
        return responses
