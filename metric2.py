import numpy as np
import cv2
from scipy.stats import entropy
from skimage.feature import local_binary_pattern
from skimage.measure import shannon_entropy
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import torch
import lpips

# -----------------------------------------------------------
# FONCTIONS DE COMPARAISON COMPLETES
# -----------------------------------------------------------

def calculate_all_metrics(original_image, enhanced_image):
    """
    Calcule toutes les métriques pour comparer une image originale et son amélioration.
    """
    assert original_image.shape[2] == 3, "L'image originale doit être en couleur (3 canaux)"
    assert enhanced_image.shape[2] == 3, "L'image améliorée doit être en couleur (3 canaux)"
    
    original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    enhanced_gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    
    original_hsv = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    enhanced_hsv = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2HSV)
    
    original_lab = cv2.cvtColor(original_image, cv2.COLOR_BGR2LAB)
    enhanced_lab = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2LAB)
    
    metrics = {}
    metrics.update(contrast_metrics(original_gray, enhanced_gray, original_lab, enhanced_lab))
    metrics.update(detail_preservation_metrics(original_gray, enhanced_gray, original_image, enhanced_image))
    metrics.update(color_naturalness_metrics(original_hsv, enhanced_hsv, original_lab, enhanced_lab))
    metrics.update(noise_metrics(original_gray, enhanced_gray, original_image, enhanced_image))
    
    return metrics

# -----------------------------------------------------------
# CONTRASTE
# -----------------------------------------------------------
def contrast_metrics(original_gray, enhanced_gray, original_lab, enhanced_lab):
    def local_contrast(img, window_size=7):
        mean_img = cv2.boxFilter(img, -1, (window_size, window_size))
        mean_squared_img = cv2.boxFilter(img**2, -1, (window_size, window_size))
        return np.mean(np.sqrt(mean_squared_img - mean_img**2))

    def calculate_eme(img, block_size=8):
        h, w = img.shape
        eme_sum = 0
        num_blocks = 0
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = img[i:i+block_size, j:j+block_size]
                min_val = np.min(block)
                max_val = np.max(block)
                if min_val > 0:
                    eme_sum += 20 * np.log10(max_val / min_val)
                    num_blocks += 1
        return eme_sum / max(1, num_blocks)

    return {
        'global_contrast': np.std(enhanced_gray),
        'global_contrast_improvement': np.std(enhanced_gray) / np.std(original_gray),
        'local_contrast': local_contrast(enhanced_gray),
        'local_contrast_improvement': local_contrast(enhanced_gray) / local_contrast(original_gray),
        'luminance_contrast': np.std(enhanced_lab[:,:,0]),
        'luminance_contrast_improvement': np.std(enhanced_lab[:,:,0]) / np.std(original_lab[:,:,0]),
        'eme': calculate_eme(enhanced_gray),
        'eme_improvement': calculate_eme(enhanced_gray) / max(0.001, calculate_eme(original_gray))
    }

# -----------------------------------------------------------
# DETAILS
# -----------------------------------------------------------
def detail_preservation_metrics(original_gray, enhanced_gray, original_color, enhanced_color):
    def gradient_magnitude(img):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(sobelx**2 + sobely**2)

    def compute_mscn(img, window_size=7):
        mu = cv2.GaussianBlur(img, (window_size, window_size), 1.0)
        mu_sq = cv2.GaussianBlur(img**2, (window_size, window_size), 1.0)
        sigma = np.sqrt(abs(mu_sq - mu**2))
        sigma = np.maximum(sigma, 1e-8)
        return (img - mu) / sigma

    def ssim_index(img1, img2):
        C1 = (0.01 * 255)**2
        C2 = (0.03 * 255)**2
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())
        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1**2
        mu2_sq = mu2**2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return np.mean(ssim_map)

    grad_mag_original = gradient_magnitude(original_gray)
    grad_mag_enhanced = gradient_magnitude(enhanced_gray)
    laplacian_original = cv2.Laplacian(original_gray, cv2.CV_64F)
    laplacian_enhanced = cv2.Laplacian(enhanced_gray, cv2.CV_64F)
    entropy_original = shannon_entropy(original_gray)
    entropy_enhanced = shannon_entropy(enhanced_gray)
    mscn_original = compute_mscn(original_gray)
    mscn_enhanced = compute_mscn(enhanced_gray)

    return {
        'gradient_magnitude': np.mean(grad_mag_enhanced),
        'gradient_magnitude_ratio': np.mean(grad_mag_enhanced) / np.mean(grad_mag_original),
        'laplacian_variance': np.var(laplacian_enhanced),
        'laplacian_variance_ratio': np.var(laplacian_enhanced) / max(0.001, np.var(laplacian_original)),
        'entropy': entropy_enhanced,
        'entropy_improvement': entropy_enhanced / entropy_original,
        'mscn_variance': np.var(mscn_enhanced),
        'detail_preservation_index': ssim_index(original_gray, enhanced_gray)
    }

# -----------------------------------------------------------
# NATURALITE DES COULEURS
# -----------------------------------------------------------
def color_naturalness_metrics(original_hsv, enhanced_hsv, original_lab, enhanced_lab):
    def color_harmony_measure(hsv_img):
        hist_hue = cv2.calcHist([hsv_img], [0], None, [36], [0, 180])
        hist_hue = hist_hue / np.sum(hist_hue)
        return -np.sum(hist_hue * np.log2(hist_hue + 1e-7))

    def color_naturalness_index(lab_img):
        natural_mean_a, natural_mean_b = 0, 0
        natural_std_a, natural_std_b = 15, 15
        mean_a = np.mean(lab_img[:,:,1])
        mean_b = np.mean(lab_img[:,:,2])
        std_a = np.std(lab_img[:,:,1])
        std_b = np.std(lab_img[:,:,2])
        mean_dist = np.sqrt((mean_a - natural_mean_a)**2 + (mean_b - natural_mean_b)**2)
        std_dist = np.abs(std_a - natural_std_a) + np.abs(std_b - natural_std_b)
        return np.exp(-(mean_dist/30 + std_dist/30))

    def histogram_distance(hist1, hist2):
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)

    return {
        'saturation_mean': np.mean(enhanced_hsv[:,:,1]),
        'saturation_change': np.mean(enhanced_hsv[:,:,1]) / np.mean(original_hsv[:,:,1]),
        'saturation_variance': np.var(enhanced_hsv[:,:,1]),
        'saturation_variance_ratio': np.var(enhanced_hsv[:,:,1]) / max(0.001, np.var(original_hsv[:,:,1])),
        'color_contrast': np.sqrt(np.std(enhanced_lab[:,:,1])**2 + np.std(enhanced_lab[:,:,2])**2),
        'color_contrast_change': np.sqrt(np.std(enhanced_lab[:,:,1])**2 + np.std(enhanced_lab[:,:,2])**2) /
                                 max(0.001, np.sqrt(np.std(original_lab[:,:,1])**2 + np.std(original_lab[:,:,2])**2)),
        'color_harmony': color_harmony_measure(enhanced_hsv),
        'color_harmony_change': color_harmony_measure(enhanced_hsv) / max(0.001, color_harmony_measure(original_hsv)),
        'color_naturalness_index': color_naturalness_index(enhanced_lab),
        'color_naturalness_change': color_naturalness_index(enhanced_lab) / max(0.001, color_naturalness_index(original_lab)),
        'color_distribution_distance': (histogram_distance(
            cv2.calcHist([original_lab], [1], None, [256], [-128, 127]),
            cv2.calcHist([enhanced_lab], [1], None, [256], [-128, 127])
        ) + histogram_distance(
            cv2.calcHist([original_lab], [2], None, [256], [-128, 127]),
            cv2.calcHist([enhanced_lab], [2], None, [256], [-128, 127])
        )) / 2
    }

# -----------------------------------------------------------
# BRUIT
# -----------------------------------------------------------
def noise_metrics(original_gray, enhanced_gray, original_color, enhanced_color):
    def estimate_noise_in_homogeneous_regions(img, threshold=5):
        dx = np.abs(np.diff(img, axis=1))
        dy = np.abs(np.diff(img, axis=0))
        mask_x = np.pad(dx < threshold, ((0, 0), (0, 1)), mode='constant')
        mask_y = np.pad(dy < threshold, ((0, 1), (0, 0)), mode='constant')
        homogeneous_mask = mask_x & mask_y
        if np.sum(homogeneous_mask) > 0:
            return np.std(img[homogeneous_mask])
        else:
            return np.std(img) * 0.1

    def high_frequency_energy(img):
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        rows, cols = img.shape
        crow, ccol = rows//2, cols//2
        mask = np.ones((rows, cols), np.uint8)
        r = min(crow, ccol) // 3
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0])**2 + (y - center[1])**2 <= r*r
        mask[mask_area] = 0
        return np.sum(magnitude * mask) / np.sum(mask)

    def calculate_lbp_roughness(img, radius=1, n_points=8):
        lbp = local_binary_pattern(img, n_points, radius, method='uniform')
        n_bins = n_points + 2
        hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
        return entropy(hist)

    def estimate_snr(img, noise_estimate):
        signal_power = np.mean(img**2)
        noise_power = noise_estimate**2
        return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100

    noise_original = estimate_noise_in_homogeneous_regions(original_gray)
    noise_enhanced = estimate_noise_in_homogeneous_regions(enhanced_gray)

    return {
        'noise_level': noise_enhanced,
        'noise_amplification': noise_enhanced / max(0.001, noise_original),
        'high_frequency_energy': high_frequency_energy(enhanced_gray),
        'high_frequency_ratio': high_frequency_energy(enhanced_gray) / max(0.001, high_frequency_energy(original_gray)),
        'lbp_roughness': calculate_lbp_roughness(enhanced_gray),
        'lbp_roughness_ratio': calculate_lbp_roughness(enhanced_gray) / max(0.001, calculate_lbp_roughness(original_gray)),
        'snr_estimated': estimate_snr(enhanced_gray, noise_enhanced),
        'snr_change': estimate_snr(enhanced_gray, noise_enhanced) - estimate_snr(original_gray, noise_original)
    }

# -----------------------------------------------------------
# COMPARAISON GLOBALE
# -----------------------------------------------------------
def compare_algorithms(original_image, results_dict):
    comparison = {}
    for algo_name, result_image in results_dict.items():
        metrics = calculate_all_metrics(original_image, result_image)
        comparison[algo_name] = metrics
    return comparison

# -----------------------------------------------------------
# METRIQUES PERCEPTUELLES
# -----------------------------------------------------------
def charger_image_cv2(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image non trouvée : {path}")
    return image

def calculer_psnr(img1, img2):
    return cv2.PSNR(img1, img2)

def calculer_ssim(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score




# Chemins vers les images
chemin_originale = "emma.jpg"
chemin_degradee = "images_transformees/emma_faible_luminosite.jpg"
chemin_restauree = "images_restaurees/resultats/lime_result.jpg"

# Chargement des images
img_originale = charger_image_cv2(chemin_originale)
img_degradee = charger_image_cv2(chemin_degradee)
img_restauree = charger_image_cv2(chemin_restauree)

# 🔹 MÉTRIQUES SANS RÉFÉRENCE : dégradée vs restaurée
print("\n--- MÉTRIQUES SANS RÉFÉRENCE (input vs output) ---")
resultats_nr = calculate_all_metrics(img_degradee, img_restauree)
for cle, valeur in resultats_nr.items():
    print(f"{cle}: {valeur:.4f}")

# 🔹 MÉTRIQUES AVEC RÉFÉRENCE : originale vs restaurée
print("\n--- MÉTRIQUES AVEC RÉFÉRENCE (ground-truth vs output) ---")
print(f"PSNR: {calculer_psnr(img_originale, img_restauree):.2f} dB")
print(f"SSIM: {calculer_ssim(img_originale, img_restauree):.4f}")
