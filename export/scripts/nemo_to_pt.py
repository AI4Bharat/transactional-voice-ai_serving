# python nemo_to_pt.py ../models/ta/ta.nemo ../data/ta.wav

import torch
import nemo.collections.asr as nemo_asr
import sys
import numpy as np
import os
from nemo.collections.asr.modules.audio_preprocessing import AudioToMelSpectrogramPreprocessor
import hydra
from omegaconf import open_dict
import librosa
import time
from pyctcdecode import build_ctcdecoder

path = sys.argv[1]
audio_filepath = sys.argv[2]

feat_extractor_path = f"{os.path.dirname(path)}/feature_extractor.pt"
acoustic_model_path = path.replace(".nemo", ".pt")

asr_model = nemo_asr.models.EncDecCTCModelBPE.restore_from(path, map_location=torch.device("cuda"))
asr_model.eval()
preprocessor = asr_model.preprocessor

old_preprocessor = preprocessor
new_config = asr_model.cfg.preprocessor
with open_dict(new_config):
    new_config.use_torchaudio = True
    
# Instantiate an instance that uses torchaudio on the backend
new_preprocessor: AudioToMelSpectrogramPreprocessor = hydra.utils.instantiate(config=new_config)
new_preprocessor.eval()

with torch.autocast(device_type="cuda", enabled=False):
    new_preprocessor.export(feat_extractor_path)
    asr_model.export(acoustic_model_path)

model = torch.jit.load(acoustic_model_path)
audio, _ = librosa.load(path=audio_filepath, sr=16000)

# [1, T_audio]
audio_tensor = torch.tensor(audio[np.newaxis, :], dtype=torch.float32)
audio_len_tensor = torch.tensor([audio.shape[0]], dtype=torch.int32)

# -- checking runtime efficiency of new_preprocessor vs preprocessor
st1 = time.time()
features, seq_len = preprocessor(input_signal=audio_tensor.cuda(), length=audio_len_tensor.cuda())
st2 = time.time()
elapsed1 = st2 - st1
features, seq_len = new_preprocessor(input_signal=audio_tensor, length=audio_len_tensor)
elapsed2 = time.time() - st2
print(elapsed1, elapsed2)

logprob = model(features.cuda(), seq_len.cuda())
decoder = build_ctcdecoder(asr_model.decoder.vocabulary)
text = decoder.decode(logprob[0].cpu().numpy()) # prediction from converted cpt

# get prediction from original nemo ckpt
logits_og = asr_model.transcribe(paths2audio_files=[audio_filepath], batch_size=1, logprobs=True)

text_og = decoder.decode(logits_og[0])
print(f"OG, {text_og} \nPT, {text} ")

print(asr_model.decoder.vocabulary)
print(len(asr_model.decoder.vocabulary))
