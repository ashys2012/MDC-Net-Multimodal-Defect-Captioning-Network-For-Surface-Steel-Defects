import subprocess

# Function to check if torchvision.metrics.map is available
def check_torchvision_map():
    # Script to check for the availability of MeanAveragePrecision in torchmetrics

    try:
        # Attempt to import MeanAveragePrecision from torchmetrics
        from torchmetrics.detection.mean_ap import MeanAveragePrecision

        print("MeanAveragePrecision is available in your torchmetrics installation.")
    except ImportError as e:
        print(f"MeanAveragePrecision is not available: {e}")


# Checking torchvision.metrics.map
print(check_torchvision_map())

import torch
import torchvision

# Print PyTorch version
print("PyTorch version:", torch.__version__)

# Print torchvision version
print("torchvision version:", torchvision.__version__)

# If CUDA is available, print CUDA version
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)

    # cuDNN version can be checked if CUDA is available
    print("cuDNN version:", torch.backends.cudnn.version())

else:
    print("no cuda")
