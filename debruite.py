import cv2
import numpy as np
import matplotlib.pyplot as plt

# Chargement d'une image bruitée
img = cv2.imread('image_bruitee.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Application du filtre bilatéral
img_denoised = cv2.bilateralFilter(img, d=5, sigmaColor=50, sigmaSpace=50)
dst_color = cv2.fastNlMeansDenoisingColored(img, None, h=8, hColor=10, templateWindowSize=3, searchWindowSize=21)

# Affichage
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('Image Bruitée')
plt.imshow(img)
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Image Débruitée')
#plt.imshow(img_denoised)
plt.imshow(dst_color)
plt.axis('off')
plt.show()
