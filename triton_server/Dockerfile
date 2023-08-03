FROM nvcr.io/nvidia/tritonserver:22.12-py3
RUN apt-get update && apt-get install -y python3.8-venv libsndfile1 build-essential libssl-dev swig
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONDONTWRITEBYTECODE=1
RUN pip3 install --no-cache-dir pyctcdecode webrtcvad setuptools==59.5.0 transformers torch --extra-index-url https://download.pytorch.org/whl/cu116
RUN cd /tmp && wget https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1-linux-x86_64.tar.gz &&\
    tar -zxvf cmake-3.25.1-linux-x86_64.tar.gz 
RUN git clone https://github.com/nvidia-riva/riva-asrlib-decoder &&\
    cd riva-asrlib-decoder && git submodule update --init --recursive &&\
    mkdir build && cd build &&\
    /tmp/cmake-3.25.1-linux-x86_64/bin/cmake -DRIVA_ASRLIB_BUILD_PYTHON_BINDINGS=NO .. &&\
    make -j 2
COPY scripts/swig /home/swig
RUN cd /home/swig && bash setup.sh
RUN python -m pip install git+https://github.com/huggingface/optimum.git#egg=optimum[onnxruntime-gpu] 
RUN pip install --no-cache-dir --force-reinstall "faster-whisper @ https://github.com/guillaumekln/faster-whisper/archive/3adcc12d0f91369446a624e33185c555facc8ed2.tar.gz" joblib charset-normalizer==3.1.0
RUN pip uninstall -y onnxruntime && pip install --no-cache-dir onnxruntime-gpu==1.13.1 numpy==1.23.1
RUN git clone https://github.com/AI4Bharat/indic-punct.git && cd indic-punct && git checkout deployment && bash install.sh && pip install --no-cache-dir pynini && python setup.py bdist_wheel && pip install -e .
RUN pip install pyyaml accelerate protobuf https://github.com/kpu/kenlm/archive/master.zip transformers -U
RUN pip install protobuf==3.20 
RUN cd indic-punct && git pull && git checkout transactional-voice-ai
CMD tritonserver --model-repository=/models/model_repository
EXPOSE 8001
EXPOSE 8002
EXPOSE 8003