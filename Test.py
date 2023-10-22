import torch
import torch.nn as nn
from skimage import measure
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt


def prediction(X, model, j, area_threshold=100, level_threshold=0.8, save=False):
    rgb_mean = torch.Tensor([0.485, 0.456, 0.406])
    rgb_std = torch.Tensor([0.229, 0.224, 0.225])
    function = nn.Softmax(1)

    pred = function(model(X)).detach().numpy()

    binary_image = pred[j, 1, ::]

    contours = measure.find_contours(binary_image, level_threshold)

    output_image = (X[j].permute(1, 2, 0) * rgb_std + rgb_mean).numpy()

    average_values = []
    titles_pos = []

    for contour in contours:
        contour = contour.astype(int)
        if cv2.contourArea(contour) >= area_threshold:
            contour_rotated = np.fliplr(contour)
            titles_pos.append(
                (np.max(contour_rotated[:, 0]), np.max(contour_rotated[:, 1]))
            )
            mask = np.zeros(binary_image.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour_rotated], -1, 1, thickness=cv2.FILLED)
            average_values.append(np.mean(binary_image[mask == 1]))
            output_image[:, :, 1] += mask * 0.3
            for point in contour_rotated:
                x, y = point
                output_image[y, x] = [0, 1, 0]

    plt.imshow(np.clip(output_image, 0, 1));
    for i, (x, y) in enumerate(titles_pos):
        plt.text(x + 5, y + 5, f'Gun: {int(average_values[i] * 100)}%', backgroundcolor='green', c='w')
    plt.axis('off')
    plt.savefig(f'tests/image{j+1}-'+''.join(random.choice('0123456789') for _ in range(10))) if save else plt.show();
    plt.close()
