FROM nvcr.io/nvidia/pytorch:19.12-py3

COPY ./requirements.txt /requirements.txt

WORKDIR /

RUN pip install --no-cache-dir -r requirements.txt
RUN rm requirements.txt

RUN git clone https://github.com/saturnin13/SATs-graph-neural-network-solver.git
RUN git config --global user.email "saturnin.13@hotmail.com"
RUN git config --global user.name "saturnin13"
RUN git config --global credential.helper store

#RUN git clone https://github.com/NVIDIA/apex
#RUN cd apex && pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

WORKDIR /SATs-graph-neural-network-solver



