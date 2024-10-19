import diffusers
import torch

LORA_WEIGHTS = "onstage3890/maya_model_v1_lora"

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

def load_model():
    pipeline = diffusers.AutoPipelineForText2Image.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=dtype
    )
    pipeline.load_lora_weights(
        LORA_WEIGHTS, weight_name="pytorch_lora_weights.safetensors"
    )
    pipeline.to(device)
    return pipeline
