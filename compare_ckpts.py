# import torch

# ckpt1 = torch.load('ckpts/grounding_dino_swin-b_pretrain_all-f9818a7c.pth')
# ckpt2 = torch.load('work_dirs/grounding_dino_bbyy_swin-b_seg_cityscapes_anomaly/iter_5000.pth')

# for k, v in ckpt1['state_dict'].items():
#     try:
#         if not (v == ckpt2['state_dict'][k]).all():
#             print(k)
#     except:
#         pass


import torch
import copy


ckpt1 = torch.load('work_dirs/grounding_dino_bbyy_swin-b_finetune_obj365/iter_38038.pth')
ckpt2 = torch.load('work_dirs/grounding_dino_bbyy_swin-b_seg_cityscapes_anomaly/iter_5000.pth')

ckpt3 = copy.deepcopy(ckpt2)
for k, v in ckpt1['state_dict'].items():
    if 'bbyy' in k:
        ckpt3['state_dict'][k] = v

torch.save(ckpt3, 'work_dirs/grounding_dino_bbyy_swin-b_seg_cityscapes_anomaly/iter_5000.pth')
