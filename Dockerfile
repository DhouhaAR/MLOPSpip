FROM jupyter/scipy-notebook

RUN mkdir my-model processed_data results
ENV MODEL_DIR=/home/jovyan/my-model
ENV MODEL_FILE_LDA=clf_lda.joblib
ENV MODEL_FILE_NN=clf_nn.joblib
ENV PROCESSED_DATA_DIR=/home/jovyan/processed_data
ENV RESULTS_DIR=/home/jovyan/results


COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt 

COPY train.csv ./train.csv
COPY test.csv ./test.csv
COPY test.json ./test.json

copy inference.py ./inference.py
COPY train.py ./train.py
COPY api.py ./api.py


RUN python3 train.py
