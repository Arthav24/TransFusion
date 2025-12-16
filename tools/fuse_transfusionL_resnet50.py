import torch
pts = torch.load('/home/project/TransFusion/work_dirs/transfusion_nusc_pillar_L/epoch_20.pth',  map_location='cpu')

img = torch.load('/home/project/TransFusion/checkpoints/resnet50.pth',  map_location='cpu')

def extract_state(d):
    # If checkpoint already contains a 'state_dict' key
    if "state_dict" in d:
        return d["state_dict"]

    # If checkpoint contains only weights (no wrapper)
    if all(isinstance(k, str) for k in d.keys()):
        return d

    # Try common alternative keys
    for key in ["model", "weights", "state", "params"]:
        if key in d:
            return d[key]

    raise KeyError("No state_dict found in checkpoint. Keys: " + str(d.keys()))

img_state = extract_state(img)
pts_state = extract_state(pts)

new_model = {"state_dict": pts_state.copy()}

for k, v in img_state.items():
    if 'backbone' in k or 'neck' in k:
        new_model["state_dict"]['img_'+k] = v

torch.save(new_model, "fusion_model_AMP.pth")