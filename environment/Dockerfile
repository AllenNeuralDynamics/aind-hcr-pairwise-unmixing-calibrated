# Code Ocean environment for aind-hcr-pairwise-unmixing-calibrated.
#
# Deliberately minimal: this package needs numpy/scipy/pandas/pyarrow/sklearn/anndata
# and nothing else. It does NOT depend on aind-spot-spectral-unmixing, so none of the
# upstream engine's git dependencies appear here.
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/mambaforge3:23.1.0-4-python3.10.12-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.1 \
    pandas==2.2.2 \
    pyarrow==16.1.0 \
    scikit-learn==1.5.0 \
    anndata==0.10.8 \
    h5py==3.11.0

# The package is NOT pip-installed here. Its source lives in the capsule at
# /root/capsule/src, which is mounted at run time -- installing a build-time copy would
# go stale the moment the code is edited. code/run_capsule.py puts that path on
# sys.path instead, so an edit takes effect on the next run with no rebuild.
