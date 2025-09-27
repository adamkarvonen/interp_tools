# %%

# upload a folder

from huggingface_hub import HfApi, upload_folder

api = HfApi()

# repo_id = "qwen3-8b-layer0-decoder-train-layers-9-18-27"
repo_id = "checkpoints_sst2_layer_9_offset_-4_None_checkpoints_larger_dataset_decoder_final"
username = "adamkarvonen"

folder = f"{repo_id}/final"

# create repo if it doesn't exist
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

api.upload_folder(folder_path=folder, repo_id=f"{username}/{repo_id}", repo_type="model")

# %%

# download a folder


def download_hf_folder(repo_id, folder_path, local_dir):
    """
    Download a specific folder from a Hugging Face repository.

    Args:
        repo_id: The repository ID (e.g., "adamkarvonen/loras")
        folder_path: The path to the folder in the repo (e.g., "model_lora_Qwen_Qwen3-8B_evil_claude37")
        local_dir: The local directory to save the files (e.g., "model_lora")
    """

    # Method 1: Using snapshot_download with allow_patterns
    # This is the most efficient way to download a specific folder
    from pathlib import Path

    from huggingface_hub import hf_hub_download, snapshot_download

    try:
        # Create the local directory if it doesn't exist
        Path(local_dir).mkdir(parents=True, exist_ok=True)

        # Download only files from the specific folder
        # The pattern matches all files within the specified folder
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=f"{folder_path}/*",  # Only download files from this folder
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # Download actual files, not symlinks
        )

        print(f"Successfully downloaded {folder_path} from {repo_id} to {local_dir}")

        # The files will be in a subfolder structure, so you might want to move them
        # to the root of your local_dir if needed

    except Exception as e:
        print(f"Error downloading folder: {e}")


repo_id = "adamkarvonen/loras"
folder_path = "model_lora_Qwen_Qwen3-8B_evil_claude37"
local_dir = "model_lora"

# Method 1: Using snapshot_download (recommended)
download_hf_folder(repo_id, folder_path, local_dir)
