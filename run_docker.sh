#!/usr/bin/env bash

cd "$(dirname "$0")" || exit 1

registry="docker.io/zqchen33"
project="rnrdp"
tag="dev"
prefix="rnrdp-$tag"

docker run -it --rm --runtime=nvidia --gpus all --network=host \
    --user "$(id -u):$(id -g)" \
    --env-file .env \
    -v "$(pwd):/workspace/$project" \
    --workdir /workspace/$project \
    --name $prefix-container \
    --entrypoint "/workspace/$project/entrypoint.sh" \
    $registry/$project:$tag
