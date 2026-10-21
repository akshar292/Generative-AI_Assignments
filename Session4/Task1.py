
import cv2
import numpy as np
import os

def forward_diffusion(image_path, steps=10):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError("Please check that input.jpg exists.")

    os.makedirs("noisy_images", exist_ok=True)

    noisy_image = image.astype(np.float32)

    for step in range(1, steps + 1):
        noise = np.random.normal(
            0, 15, image.shape
        ).astype(np.float32)

        noisy_image = np.clip(
            noisy_image + noise, 0, 255
        )

        output = noisy_image.astype(np.uint8)

        cv2.imwrite(
            f"noisy_images/step_{step}.png", output
        )

        print(f"Saved diffusion step {step}")

    return image, output

original, final_noisy = forward_diffusion("input.jpg", 10)
