import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Yolo_loss(nn.Module):
    def __init__(self, n_classes=80, n_anchors=3, device=None, batch=2):
        super(Yolo_loss, self).__init__()
        self.device = device
        self.strides = [8, 16, 32]
        image_size = 224 #608
        # image_size = 608
        self.n_classes = n_classes
        self.n_anchors = n_anchors

        self.anchors = [[12, 16], [19, 36], [40, 28], [36, 75], [76, 55], [72, 146], [142, 110], [192, 243], [459, 401]]
        self.anch_masks = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        self.ignore_thre = 0.5

        self.masked_anchors, self.ref_anchors, self.grid_x, self.grid_y, self.anchor_w, self.anchor_h = [], [], [], [], [], []

        for i in range(3):
            all_anchors_grid = [(w / self.strides[i], h / self.strides[i]) for w, h in self.anchors]       # Computer all anchors for the respective feture map
            masked_anchors = np.array([all_anchors_grid[j] for j in self.anch_masks[i]], dtype=np.float32) # Keep only the corresponding anchors
            ref_anchors = np.zeros((len(all_anchors_grid), 4), dtype=np.float32)
            ref_anchors[:, 2:] = np.array(all_anchors_grid, dtype=np.float32)
            ref_anchors = torch.from_numpy(ref_anchors)                     # Define ref anchors with 0,0 centres([0:2]) and sizes scaled ([2:4])
            # calculate pred - xywh obj cls
            fsize = image_size // self.strides[i]                           # Factor of downsample for the respective feature map
            grid_x = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).to(device)                      # 0 1 2 3 / 0 1 2 3 /
            grid_y = torch.arange(fsize, dtype=torch.float).repeat(batch, 3, fsize, 1).permute(0, 1, 3, 2).to(device)  # 0 0 0 0 / 1 1 1 1 /
            anchor_w = torch.from_numpy(masked_anchors[:, 0]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)
            anchor_h = torch.from_numpy(masked_anchors[:, 1]).repeat(batch, fsize, fsize, 1).permute(0, 3, 1, 2).to(
                device)

            self.masked_anchors.append(masked_anchors)
            self.ref_anchors.append(ref_anchors)
            self.grid_x.append(grid_x)
            self.grid_y.append(grid_y)
            self.anchor_w.append(anchor_w)
            self.anchor_h.append(anchor_h)
            
            
def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    
    # print(torch.min(w1, w2).shape)
    
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area

            
if __name__ == "__main__":
    # yolo_loss = Yolo_loss()
    
    # print(yolo_loss.anchor_h)
    # print(yolo_loss.anchor_h[0].shape)
    # print(yolo_loss.anchor_w)
    
    anchors = torch.FloatTensor([(1,2), (3,4), (5,6)])
    # print(anchors.shape)
    gwh = torch.zeros(32,2)
    
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    print(ious.shape)
    
    for i, anchor_ious in enumerate(ious.t()):
        print(anchor_ious > 0)
        break
    
    a = torch.Tensor([1, 2, 3])
    print(a[[True,False,True]])
    
    # print(ious.max(0)[0].shape)
    
    # testT = torch.zeros(10,6)
    # testT[0,4] = 1
    # testT[1,4] = 11
    # testT[2,4] = 111
    # testT[0,5] = 2
    # testT[1,5] = 22
    # testT[2,5] = 222
    
    # targetTest = testT[:,2:6]
    # # print(targetTest.shape)
    # gwh = targetTest[:, 2:]
    # # print(gwh.shape)
    # # gw, gh = gwh.t()
    # # print(gw.shape)
    
    # print(gwh)
    # print(gwh.t())
    