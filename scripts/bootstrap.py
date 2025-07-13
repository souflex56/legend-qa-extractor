import spacy
import subprocess
import sys
import requests
from spacy.util import get_installed_models

def download_spacy_model(model_name="zh_core_web_sm"):
    """
    Checks if a SpaCy model is installed and downloads it if not.
    Provides a fallback to direct pip install if the spacy download command fails.
    """
    if model_name in get_installed_models():
        print(f"✅ SpaCy model '{model_name}' is already installed.")
        return

    print(f"ℹ️ SpaCy model '{model_name}' not found. Attempting to download...")
    
    try:
        # First, try the standard spacy download command
        subprocess.check_call([sys.executable, "-m", "spacy", "download", model_name])
        print(f"✅ Successfully downloaded '{model_name}' via spaCy.")
        return
    except subprocess.CalledProcessError as e:
        print(f"⚠️ SpaCy download command failed. Error: {e}")
        print("   Attempting fallback to direct pip install from URL...")

    # Fallback: Construct the URL and use pip
    try:
        # Get the latest version compatible with the installed spacy
        spacy_version = spacy.__version__
        # This URL points to the compatibility file. We will try to fetch it.
        compat_url = "https://raw.githubusercontent.com/explosion/spacy-models/master/compatibility.json"
        response = requests.get(compat_url)
        response.raise_for_status()
        compat = response.json()
        
        # Find the latest compatible model version
        model_versions = compat.get("spacy", {}).get(spacy_version, {}).get(model_name, [])
        if not model_versions:
            raise ValueError(f"No compatible version found for {model_name} with spacy v{spacy_version}")

        latest_version = model_versions[0]
        
        # Construct the download URL for the wheel file
        model_url = f"https://github.com/explosion/spacy-models/releases/download/{model_name}-{latest_version}/{model_name}-{latest_version}-py3-none-any.whl"
        
        print(f"   Downloading from: {model_url}")
        
        # Use pip to install from the URL
        subprocess.check_call([sys.executable, "-m", "pip", "install", model_url])
        
        print(f"✅ Successfully installed '{model_name}' via pip.")

    except Exception as fallback_e:
        print(f"❌ Fallback installation failed. Error: {fallback_e}")
        print("\n========================= MANUAL INSTALLATION REQUIRED =========================")
        print(f"Please try installing the model manually. You might need to find the correct URL for your system.")
        print("A common command is:")
        print(f"   pip install https://github.com/explosion/spacy-models/releases/download/zh_core_web_sm-3.7.0/zh_core_web_sm-3.7.0-py3-none-any.whl")
        print("If you are behind a proxy, you may need to configure it for pip:")
        print("   pip --proxy http://your-proxy-address:port install ...")
        print("================================================================================")
        sys.exit(1)

if __name__ == "__main__":
    download_spacy_model() 