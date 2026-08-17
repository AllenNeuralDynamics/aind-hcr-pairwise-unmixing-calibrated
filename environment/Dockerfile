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

# Only what the base image may not already carry. Deliberately unpinned beyond a
# floor: pinning exact versions against an unknown base image is how you get an
# unsatisfiable resolve. numpy / scipy / pandas / scikit-learn / h5py come with the
# base image.
RUN pip install -U --no-cache-dir \
    "pyarrow>=14" \
    "anndata>=0.10"

# The package is NOT pip-installed here. Its source lives in the capsule at
# /root/capsule/src, which is mounted at run time -- installing a build-time copy would
# go stale the moment the code is edited. code/run_capsule.py puts that path on
# sys.path instead, so an edit takes effect on the next run with no rebuild.
