#!/usr/bin/env bash

docker run -it \
    --rm \
    --gpus=all \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --env HSA_OVERRIDE_GFX_VERSION="10.3.0" \
    --user ubuntu \
    -v $(pwd):/app \
    rocm/dev-ubuntu-24.04:7.2.4-complete
