import torch

import interp_tools.saes.base_sae as base_sae
import interp_tools.saes.batch_topk_sae as batch_topk_sae
import interp_tools.saes.jumprelu_sae as jumprelu_sae


# Model configuration mapping
MODEL_CONFIGS = {
    "google/gemma-2-2b-it": {
        "total_layers": 26,  # Adding for reference
        "layer_mappings": {
            25: {"layer": 5},
            50: {"layer": 12},
            75: {"layer": 19},
        },
        "batch_size": 1,
        "trainer_id": 65,
    },
    "google/gemma-2-9b-it": {
        "total_layers": 40,  # Adding for reference
        "layer_mappings": {
            25: {"layer": 9},
            50: {"layer": 20},
            75: {"layer": 31},
        },
        "batch_size": 4,
        "trainer_id": 131,
    },
    "google/gemma-2-27b-it": {
        "total_layers": 44,  # Adding for reference
        "layer_mappings": {
            25: {"layer": 10},
            50: {"layer": 22},
            75: {"layer": 34},
        },
        "batch_size": 1,
        "trainer_id": 131,
    },
    "mistralai/Ministral-8B-Instruct-2410": {
        "total_layers": 36,
        "layer_mappings": {
            25: {"layer": 9},
            50: {"layer": 18},
            75: {"layer": 27},
        },
        "batch_size": 8,
        "trainer_id": 2,
    },
    "mistralai/Mistral-Small-24B-Instruct-2501": {
        "total_layers": 40,
        "layer_mappings": {
            25: {"layer": 10},
            50: {"layer": 20},
            75: {"layer": 30},
        },
        "batch_size": 4,
        "trainer_id": 2,
    },
    "Qwen/Qwen2.5-3B-Instruct": {
        "total_layers": 36,
        "layer_mappings": {
            25: {"layer": 9},
            50: {"layer": 18},
            75: {"layer": 27},
        },
        "batch_size": 16,
        "trainer_id": -1,
    },
}

# Gemma-specific width information
GEMMA_WIDTH_INFO = {
    "google/gemma-2-2b-it": {
        16: {
            25: "width_16k/average_l0_143",
            50: "width_16k/average_l0_82",
            75: "width_16k/average_l0_137",
        },
        65: {
            25: "width_65k/average_l0_105",
            50: "width_65k/average_l0_141",
            75: "width_65k/average_l0_115",
        },
    },
    "google/gemma-2-9b-it": {
        16: {
            25: "width_16k/average_l0_88",
            50: "width_16k/average_l0_91",
            75: "width_16k/average_l0_142",
        },
        131: {
            25: "width_131k/average_l0_121",
            50: "width_131k/average_l0_81",
            75: "width_131k/average_l0_109",
        },
    },
    "google/gemma-2-27b-it": {
        131: {
            25: "width_131k/average_l0_106",
            50: "width_131k/average_l0_82",
            75: "width_131k/average_l0_155",
        },
    },
}


def get_layer_info(model_name: str, layer_percent: int) -> int:
    """Get layer number for a given model and percentage."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not supported")

    layer_mappings = MODEL_CONFIGS[model_name]["layer_mappings"]
    if layer_percent not in layer_mappings:
        raise ValueError(f"Layer percent must be 25, 50, or 75, got {layer_percent}")

    mapping = layer_mappings[layer_percent]
    return mapping["layer"]


def load_gemma_2_sae(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
    layer_percent: int,
    trainer_id: int,
):
    """NOTE: trainer_id means the width"""
    assert trainer_id in [16, 65, 131]
    if model_name == "google/gemma-2-9b-it":
        repo_id = "google/gemma-scope-9b-it-res"
    elif model_name == "google/gemma-2-27b-it":
        repo_id = "google/gemma-scope-27b-pt-res"
    elif model_name == "google/gemma-2-2b-it":
        repo_id = "google/gemma-scope-2b-pt-res"
    else:
        raise ValueError(f"Model {model_name} not supported")

    layer = get_layer_info(model_name, layer_percent)

    # Retrieve width_info from the dedicated dictionary
    if (
        model_name not in GEMMA_WIDTH_INFO
        or trainer_id not in GEMMA_WIDTH_INFO[model_name]
        or layer_percent not in GEMMA_WIDTH_INFO[model_name][trainer_id]
    ):
        raise ValueError(
            f"Width info not available for {model_name} at {layer_percent}% layer with trainer {trainer_id}."
        )
    width_info = GEMMA_WIDTH_INFO[model_name][trainer_id][layer_percent]

    filename = f"layer_{layer}/{width_info}/params.npz"

    sae = jumprelu_sae.load_gemma_scope_jumprelu_sae(
        repo_id=repo_id,
        filename=filename,
        layer=layer,
        model_name=model_name,
        device=device,
        dtype=dtype,
        local_dir="downloaded_saes",
    )
    return sae
