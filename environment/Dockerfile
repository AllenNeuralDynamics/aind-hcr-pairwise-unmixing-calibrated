# Code Ocean environment for aind-hcr-pairwise-unmixing-calibrated.
#
# Deliberately minimal: this package needs numpy/scipy/pandas/pyarrow/sklearn/anndata
# and nothing else. It does NOT depend on aind-spot-spectral-unmixing, so none of the
# upstream engine's git dependencies appear here.
#
# Base image: the same one the existing HCR pairwise-unmixing capsule uses. Chosen
# because it is PROVEN to resolve in this deployment's registry -- a plausible-looking
# tag that does not exist fails the build with a confusing error. If you switch it via
# the UI environment editor, Code Ocean rewrites this file and adds its own
# "# hash:sha256:..." first line; that is expected and fine.
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/lightning-jupyterlab:latest

ARG DEBIAN_FRONTEND=noninteractive

# EVERY runtime dependency is listed explicitly, including the ones the base image
# probably already has. Two reasons:
#   * "the base image ships pandas" is an assumption about an image that can change
#     under us; when it is wrong the failure is ModuleNotFoundError at run time, long
#     after the build reported success.
#   * relying on pandas arriving transitively (via anndata or seaborn) means an
#     upstream dependency change silently removes it.
# Floors, not exact pins: exact pins against a base image we do not control invite an
# unsatisfiable resolve or a silent downgrade of something the image needs.
RUN pip install -U --no-cache-dir \
    "numpy>=1.24" \
    "scipy>=1.10" \
    "pandas>=2.0" \
    "pyarrow>=14" \
    "scikit-learn>=1.3" \
    "h5py>=3.8" \
    "anndata>=0.10"

# The package is NOT pip-installed here. Its source lives in the capsule at
# /root/capsule/src, which is mounted at run time -- installing a build-time copy would
# go stale the moment the code is edited. code/run_capsule.py puts that path on
# sys.path instead, so an edit takes effect on the next run with no rebuild.
