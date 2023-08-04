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
        self.vocab = ['<unk>', 'ा', 'े', 'र', 'ी', 'न', 'ि', 'ल', 'क', '्', '▁', 'स', 'म', 'त', '▁स', 'ो', '▁द', '▁क', 'ट', 'ं', '▁अ', 'प', '▁ब', '▁प', 'व', 'ु', 'य', '▁है', '▁म', 'ह', '▁ज', '▁व', '▁आ', 'ग', 'द', '▁ह', 'ू', 'श', '्र', 'ै', 'ब', '्य', '▁इ', 'ज', 'ड', '▁न', 'र्', '▁के', '▁ल', '▁में', 'च', 'ए', 'ज़', '▁उ', 'ख', '▁र', '▁फ', 'ों', 'ॉ', 'भ', '▁ग', 'ंग', 'ता', 'ने', '▁और', '▁का', 'ाइ', '्ट', '▁प्र', '▁को', '▁की', '▁कर', '▁हो', '▁से', '▁च', 'ध', '▁हैं', 'ई', '्स', '▁तो', '▁त', '▁थ', 'फ', 'थ', 'स्ट', '▁कि', 'न्ट', '▁भी', '▁ड', '▁वि', '▁नहीं', '▁रह', '▁टू', '▁एक', 'छ', '▁श', 'ड़', 'ण', 'ौ', 'ष', 'ँ', 'उ', 'इ', 'ठ', 'अ', '़', 'ओ', 'ऑ', 'घ', 'ढ', 'झ', 'आ', 'ृ', 'ऊ', 'ञ', 'ः', 'ऐ', 'औ', 'ऋ', 'ॅ', 'ङ', 'ऱ', 'ॆ', 'ऩ', 'ॐ', 'ऍ', 'ॊ', 'ॠ']
        self.static_hotwords_list = hotword_to_fn["entities-unique"](
                    lang="hi", base_path=base_path)
        self.hotword_weight=2.5 # TODO: Move all the static variables to the config.pbtxt file as parameters
        self.lm_path=os.path.join(base_path, "models/hi-lm.binary")
        self.alpha = 0.5
        self.beta = 0.5
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
