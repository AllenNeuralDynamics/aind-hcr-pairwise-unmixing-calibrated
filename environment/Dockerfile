# Code Ocean generates this file from .codeocean/environment.json. Edit that file (or
# the Environment tab in the UI) rather than this one -- a UI edit overwrites this
# Dockerfile and prepends its own "# hash:sha256:..." line.
#
# Shape mirrors the existing HCR pairwise-unmixing capsule, which is the proven
# arrangement in this deployment:
#   * lightning-jupyterlab base -- resolves in this registry, and postInstall's
#     code-server install is written against it
#   * git_ask_pass / registry_host_arg -- expected by the base image's build
#   * postInstall installs code-server, which the base image does NOT ship. Without it
#     the build fails with "Cannot Install Code-Server (VSCode)".
#
# Unlike the upstream engine's environment, nothing here clones or installs
# aind-spot-spectral-unmixing. This package does not depend on it; that isolation is
# the point of this repository.
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/lightning-jupyterlab:latest

ARG DEBIAN_FRONTEND=noninteractive

# No `COPY git-ask-pass /` and no git_ask_pass option: that file is injected by Code
# Ocean only when the build clones private git repos, and this one clones nothing. The
# COPY fails when the file is absent, so asking for it buys a failure mode for nothing.

# EVERY runtime dependency is listed explicitly, including ones the base image may
# already carry. "the base image ships pandas" is an assumption about an image that can
# change under us, and when it is wrong the failure is a ModuleNotFoundError at run
# time, long after the build reported success. Relying on pandas arriving transitively
# (via anndata or seaborn) has the same problem.
RUN pip install -U --no-cache-dir \
    numpy==1.26.4 \
    scipy==1.13.1 \
    pandas==2.2.2 \
    pyarrow==16.1.0 \
    scikit-learn==1.5.0 \
    h5py==3.11.0 \
    anndata==0.10.8

# No secrets mount here: unlike the upstream engine's postInstall this one clones
# nothing, so it needs no git access token and no API key. The working capsule mounts a
# "CO Token" secret that belongs to that capsule, not this one.
COPY postInstall /
RUN /postInstall

# The package itself is NOT pip-installed. Its source lives in the capsule at
# /root/capsule/src, mounted at run time -- installing a build-time copy would go stale
# the moment the code is edited. code/run_capsule.py puts that path on sys.path
# instead, so an edit takes effect on the next run with no rebuild.
