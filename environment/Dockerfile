# hash:sha256:f2dc5b79e790a9b36e1e7b5f8ed0358cf174c7403423f30abcd6352920cece02
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/lightning-jupyterlab:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    anndata==0.11.4 \
    matplotlib==3.10.9 \
    pyarrow==24.0.0 \
    scikit-learn==1.6.1

COPY postInstall /
RUN /postInstall
