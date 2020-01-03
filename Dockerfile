FROM pytorch/pytorch

COPY ./src /SATs-graph-neural-network-solver/src
COPY ./scripts /SATs-graph-neural-network-solver/scripts
COPY ./graphs /SATs-graph-neural-network-solver/graphs
COPY ./requirements.txt /SATs-graph-neural-network-solver/requirements.txt

WORKDIR /SATs-graph-neural-network-solver

RUN pip install -r requirements.txt
RUN ./scripts/setup.sh

CMD [ "python", "./src/main.py" ]
