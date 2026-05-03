import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18
from monai.transforms import Compose, Resize, ScaleIntensity, EnsureChannelFirst
import nibabel as nib
import numpy as np
import os

# --- CONFIGURATION ---
# Define the exact shape your model expects
TARGET_SHAPE = (96, 96, 96)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. MODEL ARCHITECTURE (Must match training EXACTLY) ---
class MultimodalFusionNetwork(nn.Module):
    def __init__(self, num_classes=3):
        super(MultimodalFusionNetwork, self).__init__()
        
        # We assume backbones are already part of the saved fusion weights, 
        # so we just need to define the structure to load the state_dict into.
        
        # MRI Encoder
        self.mri_model = self._build_backbone()
        
        # PET Encoder
        self.pet_model = self._build_backbone()
        
        # Demographics Encoder
        self.demo_net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )
        
        # Fusion Head
        self.fusion = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 + 512 + 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def _build_backbone(self):
        # Initialize standard ResNet3D-18
        model = r3d_18(weights=None) 
        
        # Modify Input to 1 Channel (Grayscale for Medical Scans)
        old_layer = model.stem[0]
        new_layer = nn.Conv3d(1, old_layer.out_channels, old_layer.kernel_size, 
                              old_layer.stride, old_layer.padding, bias=False)
        model.stem[0] = new_layer
        
        # Remove original classifier head to return 512 features
        model.fc = nn.Identity()
        return model

    def forward(self, mri, pet, demo):
        # Handle MRI
        if mri is not None:
            mri_feat = self.mri_model(mri)
        else:
            # Zero padding if missing
            batch_size = pet.shape[0] if pet is not None else demo.shape[0]
            mri_feat = torch.zeros((batch_size, 512)).to(demo.device)

        # Handle PET
        if pet is not None:
            pet_feat = self.pet_model(pet)
        else:
            batch_size = mri.shape[0] if mri is not None else demo.shape[0]
            pet_feat = torch.zeros((batch_size, 512)).to(demo.device)

        # Handle Demographics
        demo_feat = self.demo_net(demo)
        
        # Combine
        combined = torch.cat((mri_feat, pet_feat, demo_feat), dim=1)
        
        # Predict
        output = self.fusion(combined)
        return output

# --- 2. TRANSFORMS (Validation Logic) ---
# We use Monai to ensure the input scan looks exactly like the training data
inference_transforms = Compose([
    ScaleIntensity(),             # Normalize 0-1
    Resize(TARGET_SHAPE),         # Resize to (96, 96, 96)
])

# --- 3. HELPER FUNCTIONS ---

def load_model(weights_path):
    """Loads the model and weights onto CPU/GPU"""
    print(f"Loading AI Model from {weights_path}...")
    try:
        model = MultimodalFusionNetwork()
        # map_location ensures it works even if you trained on GPU but deploy on CPU
        state_dict = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval() # Set to evaluation mode
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"CRITICAL ERROR: Could not load model. {e}")
        return None

def process_scan(file_path):
    """
    Reads a NIfTI (.nii) file and converts it to a Tensor.
    Returns: Tensor of shape (1, 1, 96, 96, 96) or None if failed.
    """
    try:
        # Load NIfTI file
        img = nib.load(file_path).get_fdata()
        
        # Convert to Tensor (Add Channel Dimension: 1, H, W, D)
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        
        # Apply Transforms (Resize/Scale)
        img_tensor = inference_transforms(img_tensor)
        
        # Add Batch Dimension (1, 1, 96, 96, 96)
        img_tensor = img_tensor.unsqueeze(0)
        
        return img_tensor.to(DEVICE)
    except Exception as e:
        print(f"Error processing scan {file_path}: {e}")
        return None

def get_prediction(model, mri_path, pet_path, age, gender):
    """
    Main Interface function.
    Args:
        mri_path: Path to saved MRI file (or None)
        pet_path: Path to saved PET file (or None)
        age: Integer (e.g., 72)
        gender: String ("Male", "Female")
    """
    classes = ['CN', 'MCI', 'AD']
    
    # 1. Prepare Demographics
    # Normalize Age (divide by 100 as per training)
    age_norm = float(age) / 100.0
    # Encode Sex (1 for Female, 0 for Male/Other as per training logic)
    sex_code = 1.0 if str(gender).lower().startswith('f') else 0.0
    
    demo_tensor = torch.tensor([[age_norm, sex_code]], dtype=torch.float32).to(DEVICE)

    # 2. Process Scans
    mri_tensor = process_scan(mri_path) if mri_path else None
    pet_tensor = process_scan(pet_path) if pet_path else None

    # 3. Run Inference
    with torch.no_grad():
        logits = model(mri_tensor, pet_tensor, demo_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0] # Get probabilities
        pred_idx = np.argmax(probs) # Get highest probability index

    # 4. Format Result
    result = {
        "prediction": classes[pred_idx],
        "confidence": round(float(probs[pred_idx]) * 100, 2),
        "probabilities": {
            "CN": round(float(probs[0]) * 100, 1),
            "MCI": round(float(probs[1]) * 100, 1),
            "AD": round(float(probs[2]) * 100, 1)
        }
    }
    
    return result