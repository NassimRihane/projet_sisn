import cv2
import numpy as np
import os

def ajouter_bruit_gaussien(image, mean=0, sigma=25):
    bruit = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + bruit, 0, 255).astype(np.uint8)

def appliquer_flou_gaussien(image, ksize=(7, 7), sigma=0):
    return cv2.GaussianBlur(image, ksize, sigma)

def reduire_contraste(image, pourcentage):
    facteur = pourcentage / 100.0
    image_float = image.astype(np.float32)
    image_contraste = facteur * (image_float - 128) + 128
    return np.clip(image_contraste, 0, 255).astype(np.uint8)

def reduire_luminosite(image, facteur=0.4):
    image_float = image.astype(np.float32)
    image_lum = image_float * facteur
    return np.clip(image_lum, 0, 255).astype(np.uint8)

def distorsion_couleur(image, delta_teinte=10, facteur_saturation=1.5):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + delta_teinte) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * facteur_saturation, 0, 255)
    image_modifiee = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return image_modifiee

def main():
    chemin_image = "exemples/emma.jpg"
    nom_base = os.path.splitext(os.path.basename(chemin_image))[0]

    image = cv2.imread(chemin_image)
    if image is None:
        print(f"Erreur : l'image '{chemin_image}' n'a pas été trouvée. Vérifie le chemin.")
        return

    dossier_sortie = "images_transformees"
    os.makedirs(dossier_sortie, exist_ok=True)

    # 1. Bruit Gaussien
    image_bruitee = ajouter_bruit_gaussien(image)
    cv2.imwrite(os.path.join(dossier_sortie, f"{nom_base}_bruitee.jpg"), image_bruitee)

    # 2. Flou Gaussien
    image_floue = appliquer_flou_gaussien(image)
    cv2.imwrite(os.path.join(dossier_sortie, f"{nom_base}_floue.jpg"), image_floue)

    # 3. Réduction de contraste
    image_contraste_50 = reduire_contraste(image, 50)
    cv2.imwrite(os.path.join(dossier_sortie, f"{nom_base}_contraste_50.jpg"), image_contraste_50)

    # 4. Distorsion couleur
    image_couleur = distorsion_couleur(image, delta_teinte=20, facteur_saturation=1.5)
    cv2.imwrite(os.path.join(dossier_sortie, f"{nom_base}_couleur_modifiee.jpg"), image_couleur)

    # 5. Compression JPEG
    dossier_compression = os.path.join(dossier_sortie, "compressions")
    os.makedirs(dossier_compression, exist_ok=True)
    cv2.imwrite(os.path.join(dossier_compression, f"{nom_base}_compression_q50.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 50])

    # 6. Transformation complète (sans luminosité)
    image_finale = ajouter_bruit_gaussien(image)
    image_finale = appliquer_flou_gaussien(image_finale)
    image_finale = reduire_contraste(image_finale, 50)
    image_finale = distorsion_couleur(image_finale, delta_teinte=20, facteur_saturation=1.5)
    cv2.imwrite(os.path.join(dossier_sortie, f"{nom_base}_tout_transformee.jpg"), image_finale)
    cv2.imwrite(os.path.join(dossier_sortie, f"{nom_base}_tout_transformee_q50.jpg"), image_finale, [cv2.IMWRITE_JPEG_QUALITY, 50])

    # 7. Réduction de luminosité
    image_faible_lum = reduire_luminosite(image, facteur=0.05)
    cv2.imwrite(os.path.join(dossier_sortie, f"{nom_base}_faible_luminosite.jpg"), image_faible_lum)

    # 8. Transformation complète avec faible luminosité
    image_finale_lum = reduire_luminosite(image, facteur=0.3)
    image_finale_lum = ajouter_bruit_gaussien(image_finale_lum)
    image_finale_lum = appliquer_flou_gaussien(image_finale_lum)
    image_finale_lum = reduire_contraste(image_finale_lum, 50)
    image_finale_lum = distorsion_couleur(image_finale_lum, delta_teinte=20, facteur_saturation=1.5)
    cv2.imwrite(os.path.join(dossier_sortie, f"{nom_base}_tout_transformee_faible_lum.jpg"), image_finale_lum)

    print("✅ Toutes les images ont été enregistrées dans :", dossier_sortie)

    # Affichage (optionnel)
    cv2.imshow("Originale", image)
    cv2.imshow("Tout transformée (faible lum)", image_finale_lum)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
