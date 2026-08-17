# Code Ocean generates this file from .codeocean/environment.json. Edit that file (or the
# Environment tab) rather than this one -- a UI edit overwrites this Dockerfile and
# prepends its own "# hash:sha256:..." line.
#
# This mirrors the existing HCR pairwise-unmixing capsule as closely as possible, because
# that configuration is PROVEN to build in this deployment. Deviations from it are the
# thing most likely to break a build, and each one below is deliberate.
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/lightning-jupyterlab:latest

ARG DEBIAN_FRONTEND=noninteractive

# Only two packages, at the versions the working capsule uses.
#
# numpy / scipy / pandas / scikit-learn / h5py are deliberately NOT pinned or installed:
#   * the base image is an ML image and already carries them;
#   * anndata's own requirements (h5py>=3.1, numpy>=1.23, pandas>=1.4, scipy>1.8) pull
#     anything missing;
#   * pinning them exactly is actively dangerous here. numpy 1.26.4 ships wheels only
#     for cp39-cp312, so if this base image is Python 3.13 that pin forces a source
#     build and the environment fails. The base image's Python version is not something
#     this file should have to know.
RUN pip install -U --no-cache-dir \
    anndata==0.11.4 \
    pyarrow==24.0.0

# code-server: the base image does NOT ship it and Code Ocean's IDE requires it. Its
# absence is what produced "Cannot Install Code-Server (VSCode)".
COPY postInstall /
RUN /postInstall

# The package itself is NOT pip-installed. Its source lives in the capsule at
# /root/capsule/src, mounted at run time -- a build-time copy would go stale the moment
# the code is edited. code/run_capsule.py puts that path on sys.path instead, so an edit
# takes effect on the next run with no rebuild.
