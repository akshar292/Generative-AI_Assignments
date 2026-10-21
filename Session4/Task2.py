
import cv2
import matplotlib.pyplot as plt

original = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
noisy = cv2.imread(
    "noisy_images/step_10.png",
    cv2.IMREAD_GRAYSCALE
)

if original is None or noisy is None:
    raise FileNotFoundError("Run Task 1 first and check the image paths.")

denoised = cv2.GaussianBlur(noisy, (5, 5), 0)

cv2.imwrite("denoised_image.png", denoised)

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(original, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(noisy, cmap="gray")
plt.title("Noisy Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(denoised, cmap="gray")
plt.title("Denoised Image")
plt.axis("off")

plt.tight_layout()
plt.show()
