import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim


def _fisher_z(r: float) -> float:
    """Fisher r-to-z transformation."""
    r = np.clip(r, -0.9999, 0.9999)
    return 0.5 * np.log((1.0 + r) / (1.0 - r))


def _inv_fisher_z(z: float) -> float:
    """Inverse Fisher z-to-r transformation."""
    return float(np.tanh(z))


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      masks: np.ndarray, clip_max: float) -> dict:
    T = min(y_true.shape[0], y_pred.shape[0], masks.shape[0])
    yt = y_true[:T]
    yp = np.maximum(y_pred[:T], 0.0)
    ms = np.max(masks[:T], axis=-1, keepdims=True)  # (T,G,G,1) any-cell mask

    # RMSE
    sq_diff = np.square(yt - yp)
    masked_rmse = float(np.sqrt(
        np.sum(sq_diff * ms) / (np.sum(ms) + 1e-12)
    ))
    unmasked_rmse = float(np.sqrt(np.mean(sq_diff)))

    # global R² 
    global_r2 = float(r2_score(yt.flatten(), yp.flatten()))

    # per-timestep R² 
    per_t_r2 = []
    for t in range(T):
        gt_flat = yt[t].flatten()
        pr_flat = yp[t].flatten()
        if np.std(gt_flat) > 1e-12:
            per_t_r2.append(float(r2_score(gt_flat, pr_flat)))
        else:
            per_t_r2.append(np.nan)

    # fixed Dice 
    # 5% of clip_max as a fixed physical threshold across all timesteps
    dice_threshold = 0.05 * clip_max if clip_max > 0 else 1e-9
    dices = []
    n_empty_skipped = 0
    for t in range(T):
        gt = yt[t, :, :, 0]
        pr = yp[t, :, :, 0]
        gb = (gt > dice_threshold).astype(float)
        pb = (pr > dice_threshold).astype(float)

        if np.sum(gb) + np.sum(pb) == 0:
            n_empty_skipped += 1
            continue

        dices.append(
            (2.0 * np.sum(gb * pb)) / (np.sum(gb) + np.sum(pb) + 1e-12)
        )

    # Spatial Correlation with Fisher z-transform
    z_corrs = []
    for t in range(T):
        gt = yt[t, :, :, 0]
        pr = yp[t, :, :, 0]
        if np.std(gt) > 1e-12 and np.std(pr) > 1e-12:
            r_val = float(pearsonr(gt.flatten(), pr.flatten())[0])
            if np.isfinite(r_val):
                z_corrs.append(_fisher_z(r_val))

    if z_corrs:
        mean_z = float(np.mean(z_corrs))
        spatial_corr = _inv_fisher_z(mean_z)
    else:
        spatial_corr = 0.0

    # SSIM 
    ssims_v = []
    n_ssim_skipped = 0
    fixed_data_range = float(clip_max) if clip_max > 0 else 1.0
    for t in range(T):
        gt = yt[t, :, :, 0]
        pr = yp[t, :, :, 0]
        dr = float(np.max(gt) - np.min(gt))
        if dr < 1e-12:
            n_ssim_skipped += 1
            continue
        ssims_v.append(float(ssim(gt, pr, data_range=fixed_data_range)))

    return {
        "Global_R2":            global_r2,
        "Per_Timestep_R2":      per_t_r2,
        "Masked_RMSE":          masked_rmse,
        "Unmasked_RMSE":        unmasked_rmse,
        "Avg_Dice":             float(np.mean(dices)) if dices else 0.0,
        "Dice_Empty_Skipped":   n_empty_skipped,
        "Spatial_Correlation":  spatial_corr,
        "SSIM":                 float(np.mean(ssims_v)) if ssims_v else 0.0,
        "SSIM_Skipped_Frames":  n_ssim_skipped,
    }

def denormalize(scaled: np.ndarray, clip_max: float) -> np.ndarray:
    """convert from [-1, 1] scaled domain back to physical units."""
    return (np.asarray(scaled, dtype=np.float64) + 1.0) / 2.0 * clip_max
