docker run --gpus all -e HF_TOKEN=$HF_TOKEN -e NO_TORCH_COMPILE=1 --net host -it --rm  nv_persona:latest
