import cv2
import numpy as np
from scipy.sparse import spdiags, eye, linalg

def estimate_initial_light(img):
    img = img.astype(np.float32) / 255.0
    return np.max(img, axis=2)

def compute_weights(L, eps=1e-4):
    """Calcule les poids horizontaux et verticaux avec padding pour éviter les erreurs de dimension"""
    h, w = L.shape
    dy = np.zeros_like(L)
    dx = np.zeros_like(L)

    dy[:-1, :] = np.diff(L, axis=0)
    dx[:, :-1] = np.diff(L, axis=1)

    Wv = np.exp(- (dy ** 2) / eps)
    Wh = np.exp(- (dx ** 2) / eps)

    return Wh, Wv


def solve_illumination(L_init, alpha=0.15, eps=1e-4):
    """Résout l'équation de propagation guidée pour raffiner la carte d'illumination"""
    h, w = L_init.shape
    N = h * w
    L = L_init.flatten()
    
    Wh, Wv = compute_weights(L_init, eps)
    dx = -alpha * Wh.flatten()
    dy = -alpha * Wv.flatten()

    N = h * w
    D = np.ones(N)
    dx_flat = dx.flatten()
    dy_flat = dy.flatten()

    diags = np.array([0, -1, 1, -w, w])
    data = np.zeros((5, N))
    data[0, :] = D
    data[1, 1:] = dx_flat[:-1]      # vers la droite
    data[2, :-1] = dx_flat[1:]      # depuis la gauche
    data[3, w:] = dy_flat[:-w]      # depuis le haut
    data[4, :-w] = dy_flat[w:]      # vers le bas

    A = spdiags(data, diags, N, N, format='csr')
    refined_L = linalg.spsolve(A, L)

    return np.clip(refined_L.reshape((h, w)), 0, 1)

def enhance(img, illum, gamma=0.7):
    img = img.astype(np.float32) / 255.0
    illum = illum[..., np.newaxis]
    illum = np.maximum(illum, 0.1)
    result = (img / illum) ** gamma
    return np.clip(result * 255, 0, 255).astype(np.uint8)

# Exemple d’utilisation
if __name__ == "__main__":
    path = "images_transformees/emma_faible_luminosite.jpg"
    img = cv2.imread(path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    L_init = estimate_initial_light(img_rgb)
    L_refined = solve_illumination(L_init)
    result = enhance(img_rgb, L_refined)

    # Sauvegarde ou affichage
    cv2.imshow("Original", img_rgb)
    cv2.imshow("LIME Enhanced", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    cv2.imwrite("resultats/lime_result.jpg", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
