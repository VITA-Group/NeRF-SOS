import cv2
import numpy as np
import os

input_path = "/ssd1/xx/projects/nerfplusplus/logs/ablation_figs/1app_0geo_clus_000174.png"
img = cv2.imread(input_path)
tmp = np.ones_like(img) * 255.
img = (tmp - img).astype(np.uint8)
cv2.imwrite(input_path, img)