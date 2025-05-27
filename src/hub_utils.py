import os
from huggingface_hub import login
from huggingface_hub import HfFolder


class HFSetup:
    @staticmethod
    def setup_cache_dirs(base_cache_dir):
        """Setup cache directories for HuggingFace"""
        os.makedirs(base_cache_dir, exist_ok=True)

        cache_dirs = {
            "transformers": os.path.join(base_cache_dir, "transformers"),
            "datasets": os.path.join(base_cache_dir, "datasets"),
            "tokens": os.path.join(base_cache_dir, "tokens"),
        }

        # Create all cache directories
        for dir_path in cache_dirs.values():
            os.makedirs(dir_path, exist_ok=True)

        # Set environment variables
        os.environ["TRANSFORMERS_CACHE"] = cache_dirs["transformers"]
        os.environ["HF_DATASETS_CACHE"] = cache_dirs["datasets"]
        os.environ["HF_HOME"] = base_cache_dir

        return cache_dirs

    @staticmethod
    def login_huggingface(token):
        """Login to HuggingFace with token"""
        try:
            # First try to set token directly
            HfFolder.save_token(token)

            # Then try login
            login(token=token)
            return True
        except Exception as e:
            print(f"Warning: HuggingFace login failed: {str(e)}")
            # Continue even if login fails
            return False
