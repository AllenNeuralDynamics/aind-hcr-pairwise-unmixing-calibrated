# hash:sha256:364ae0a237c4a114353d30eb0855dab1c85efe82b17155c75ffdf43da31862db
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/lightning-jupyterlab:latest

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    anndata==0.11.4 \
    pyarrow==24.0.0 \
    scikit-learn==1.6.1

COPY postInstall /
RUN /postInstall
