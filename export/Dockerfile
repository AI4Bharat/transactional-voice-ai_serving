# sudo docker run -it --gpus=all --ipc=host -v $PWD:/workspace/npci nemo-export bash         

FROM abhigyanr/triton-asr-nemo:v0.2
RUN pip install pyyaml accelerate protobuf https://github.com/kpu/kenlm/archive/master.zip transformers -U
RUN pip install torchaudio
