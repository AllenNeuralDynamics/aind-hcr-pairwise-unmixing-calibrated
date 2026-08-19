# hash:sha256:72b6869d5f4ee2a703baf192dd30d9f1f5619b1b57e8b9790277bf7c2193f7c6
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/lightning-jupyterlab:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    anndata==0.11.4 \
    matplotlib==3.11.0 \
    pyarrow==24.0.0 \
    scikit-learn==1.6.1

COPY postInstall /
RUN /postInstall
