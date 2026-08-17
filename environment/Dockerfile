# hash:sha256:5c3b5351ce0c4dde8274dbc54dd07c3c044c0d01a7f6519f79a1b498b98dd1ae
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/lightning-jupyterlab:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    anndata==0.11.4 \
    pyarrow==24.0.0

COPY postInstall /
RUN /postInstall
