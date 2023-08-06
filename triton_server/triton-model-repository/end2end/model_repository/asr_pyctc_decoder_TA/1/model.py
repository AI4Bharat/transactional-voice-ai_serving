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
        self.vocab = ['<unk>', 'ா', 'ி', 'ு', 'வ', 'க', '▁ப', 'ை', 'ன', 'ர', 'ன்', '்', '▁க', 'ம்', 'த', 'ே', 'ய', 'ல்', '▁அ', 'ர்', 'க்க', '▁வ', 'ல', '▁ம', 'து', 'ட', 'ப்ப', 'ம', '▁த', 'ப', '▁', 'ச', 'ட்ட', 'ண', '▁ச', '▁இ', 'ும்', 'ிய', 'ோ', '▁எ', 'ெ', 'த்த', 'ூ', 'ங்க', '▁ந', 'ழ', 'ொ', 'ரு', 'தி', 'ற', 'ள', 'த்தி', 'ந்த', 'க்கு', 'ீ', 'டி', 'டு', 'ார்', 'த்து', '▁ஆ', 'ரி', 'ற்ற', 'ட்', 'கள்', '▁உ', 'ஸ்', 'வி', 'று', 'ுள்ள', '▁மு', 'லை', 'ந்து', 'ண்ட', 'ல்ல', 'க்', 'ச்ச', 'ள்', 'ளி', 'ன்ற', '▁இரு', 'ங்கள', 'யி', '▁இந்த', '▁வி', 'ட்டு', '▁செ', '▁நா', 'யில்', 'றி', 'மா', 'ந', 'ப்', 'யா', '▁கு', 'ஜ', 'டை', '▁போ', 'ற்க', '▁தொ', '▁ர', '▁நி', 'றை', 'ப்பு', 'ண்டு', '▁ஒ', '▁செய்த', 'ஷ', 'ஐ', 'ஞ', 'ஹ', 'ஓ', 'ஃ', 'ங', 'ஊ', 'ஈ', 'எ', 'ஸ', 'ௌ', 'ஆ', 'இ', 'ஏ', 'அ', 'ஒ', 'உ', 'ஔ', 'ஶ', '௧', '௭']
        self.static_hotwords_list = hotword_to_fn["entities-unique"](
                    lang="ta", base_path=base_path)
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
