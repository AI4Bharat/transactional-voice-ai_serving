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
        self.vocab = ['<unk>', '▁', 'ର', 'ା', 'ୁ', 'ି', 'ରେ', 'େ', 'ନ', 'ବ', 'ତ', 'ୋ', '▁ପ', 'ପ', 'କ', 'ମ', 'ୀ', '▁କ', 'ସ', '▁ଆ', 'ର୍', 'କୁ', '୍', 'ନ୍', '▁ବ', '▁ଅ', 'ଳ', 'ରି', 'ଇ', 'ୟ', 'ବା', '▁ସ', 'ତି', '▁ଏ', 'ଛି', 'ଣ', 'ଲ', 'ଗ', 'ଜ', '▁ବି', '▁ସେ', '▁ମ', 'କ୍', 'ଦ', 'ଟ', '▁ପ୍ର', 'ଶ', 'ଁ', '▁ଦ', 'ରୁ', 'ଭ', '▁ଭ', 'ାଇ', 'ସ୍', 'ବେ', '▁କରି', 'ରା', 'ଏ', 'ଡ', '୍ୟ', 'ଲା', 'ଖ', '▁ନ', 'ଟି', 'ହା', 'ଙ୍କ', 'ଚ', 'ହ', 'ଫ', 'ଲି', 'ଛନ୍ତି', 'ଲେ', 'ନା', 'ଉ', '▁କି', 'ଧ', '▁ଗ', 'ଷ', 'ମ୍', 'ନି', 'ହି', '▁ତ', 'ତା', '▁ଜ', '▁ଉ', 'ହେ', '▁ବା', 'ତ୍', '▁ହ', '▁ତା', 'ମି', 'ମା', 'କି', 'ତେ', '▁କା', '▁ଶ', 'ଯ', '▁ନି', 'ଷ୍ଟ', 'ିଆ', 'ଡି', 'ୂ', 'ଥ', 'ଛ', 'କ୍ଷ', '▁ପା', 'ଆ', '▁କେ', '▁ଚ', 'ସି', '▁ହୋଇ', 'ଶ୍', 'ଟ୍', 'ପା', 'ଣି', '୍ୟା', '▁ର', '▁ଏବଂ', '▁ମା', '▁ଯେ', '▁ସା', 'ନେ', 'କା', '▁ଓ', 'ରୀ', '▁ପାଇଁ', 'ବି', 'ତ୍ର', 'ଡ଼', 'ଣ୍ଡ', '▁ଦେ', 'ନ୍ତୁ', 'ଥା', 'ଥି', '▁ମଧ୍ୟ', 'ହୁ', '▁ଖ', '▁ଚା', 'ମେ', 'ଲ୍', 'ଂ', 'ମାନେ', '▁ଏକ', '▁ସ୍', 'ନ୍ତ', 'ୃ', 'ଥିଲା', 'ଟା', 'ଥିଲେ', 'ଶି', '▁ନେଇ', '▁ଲ', '▁ଆମେ', '▁ଆଉ', 'ଧା', '▁ନା', '▁ଘ', 'ଦ୍', '▁ଇ', 'ପ୍', '▁ଆମ', '▁ରା', '▁ଏହି', '▁ପୁ', 'ନ୍ତି', '▁ଆପଣ', 'ଡ଼ି', 'ଙ୍ଗ', 'ଣ୍ଟ', '▁ହେବ', 'ନ୍ଦ', '▁ଆଜି', '▁କରିବା', 'ଅ', 'ଞ୍ଚ', '▁କରୁ', 'ଗୁ', '▁ଜା', 'ଦେ', '▁କହି', '▁ତାଙ୍କ', 'ଥିବା', 'ଙ୍କୁ', 'ଠି', '▁ମୁଁ', 'କାର', '▁ମୋ', 'ଜି', '▁କିଛି', '▁ମୁଖ୍ୟ', 'ଠ', '▁ମୁ', 'ସା', '▁ସୁ', 'ର୍ଯ୍ୟ', '▁ଆସି', '▁ରାଜ', 'ମାନଙ୍କ', 'ଠା', '▁ଦୁଇ', 'ଯିବ', 'ଯାଇ', '▁ଲୋକ', 'ଜ୍', '▁ନାହିଁ', 'ଟେ', 'ବେଳେ', '▁ଦିନ', '▁କଣ', '▁ଏହା', '▁ଦେଇ', '▁କଥା', '▁ଯୋ', '▁ସବୁ', 'ଦ୍ଧ', '▁ଯେଉଁ', '▁ଜଣ', '▁ସମ୍', 'ଯୋଗ', 'ର୍ତ୍ତ', 'ଯାଉ', '▁ଆପଣଙ୍କ', '▁ରହିଛି', 'ଯାଇଛି', '▁ଯଦି', '▁ଯାହା', '▁ବର୍ଷ', 'ଭଳି', 'ପାରିବ', '▁ରହିବ', '▁ମୁଖ୍ୟମନ୍ତ୍ରୀ', 'ଘ', '▁ଦେଖୁ', 'ଗ୍ର', '▁ହେଉଛି', 'ଝ', '▁ଦେଖି', '▁ସମୟ', 'ନ୍ଦ୍ର', '▁ସହିତ', '▁ବହୁତ', '▁ଅନୁ', 'ଓ', 'ୈ', 'ୌ', 'ଢ', 'ଞ', 'ୱ', 'ଃ', 'ଋ', 'ଊ', 'ଙ', 'ଔ', 'ଈ', 'ଐ', '଼']
        self.static_hotwords_list = hotword_to_fn["entities-unique"](
                    lang="or", base_path=base_path)
        self.hotword_weight=10 # TODO: Move all the static variables to the config.pbtxt file as parameters
        self.lm_path=None
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
