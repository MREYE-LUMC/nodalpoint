"""Generate synthetic eye biometry using the SyntEyes statistical eye model.

Ports the eye-generation logic from
https://github.com/MREYE-LUMC/demos/tree/main/demos/synteyes.

The biometric model is SyntEyes (Rozema et al.DOI 10.1167/iovs.16-20877).
The retinal component is from van Dam et al. DOI:
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

MODELDATA_PATH = Path(__file__).resolve().parent / "modeldata.json"
CORNEA_ZERNIKE_DIAMETER_MM = 6.5

# Retina model parameters (van Dam extension): mean/covariance of the joint
# (axial_length, rx, ry, rz) Gaussian, sampled conditioned on the eye's own axial length.
MU_AL_RADII = np.array([23.85, 11.89, 11.66, 10.55])
COV_AL_RADII = np.array(
    [
        [1.82, 0.56, 0.42, 0.81],
        [0.56, 0.36, 0.19, 0.18],
        [0.42, 0.19, 0.24, 0.16],
        [0.81, 0.18, 0.16, 0.48],
    ]
)


def _load_modeldata() -> dict:
    modeldata = json.loads(MODELDATA_PATH.read_text(encoding="utf-8"))
    return {key: np.array(value) for key, value in modeldata.items()}


def conditional_gaussian(
    mu: np.ndarray, cov: np.ndarray, known_indices: list[int], known_values
) -> tuple:
    """Conditional mean/covariance of a multivariate Gaussian given some known variables."""
    unknown_indices = [i for i in range(len(mu)) if i not in known_indices]

    mu_known, mu_unknown = mu[known_indices], mu[unknown_indices]
    cov_kk = cov[np.ix_(known_indices, known_indices)]
    cov_ku = cov[np.ix_(known_indices, unknown_indices)]
    cov_uk = cov[np.ix_(unknown_indices, known_indices)]
    cov_uu = cov[np.ix_(unknown_indices, unknown_indices)]
    cov_kk_inv = np.linalg.inv(cov_kk)

    conditional_mean = mu_unknown + cov_uk @ cov_kk_inv @ (known_values - mu_known)
    conditional_cov = cov_uu - cov_uk @ cov_kk_inv @ cov_ku
    return conditional_mean, conditional_cov


def zernike_index(order: int) -> np.ndarray:
    """(n, m) index pairs for all OSA/ANSI Zernike terms up to radial order `order`."""
    idx = [(n, m) for n in range(order + 1) for m in range(-n, n + 1, 2)]
    return np.array(idx, dtype=float)


def _sample_gmm_eigencornea(
    modeldata: dict, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Sample `n` points from the two-component Gaussian-mixture "eigencornea" model."""
    weights = modeldata["weights_orig"]
    mu = modeldata["mu_orig"]
    covs = (modeldata["cov_orig0"], modeldata["cov_orig1"])

    counts = rng.multinomial(n, weights)
    samples = [
        rng.multivariate_normal(mu[k], cov, size=counts[k])
        for k, cov in enumerate(covs)
    ]
    return np.vstack(samples)


def _convert_single_eigencornea(eigencornea: np.ndarray, modeldata: dict) -> dict:
    """Convert one eigencornea sample to a dict of clinical SyntEyes biometry + Zernikes."""
    conv_ec, avg_ec, lens_za = (
        modeldata["conv_ec_orig"],
        modeldata["avg_ec_orig"],
        modeldata["lens_za_orig"],
    )

    zerniken = 0.001 * eigencornea[6:18].reshape(1, -1) @ conv_ec.T
    zerniken = zerniken + avg_ec

    synteyes_array = np.concatenate([eigencornea[:6], zerniken.ravel()])

    eye = {"CCT": synteyes_array[96]}
    eye["ACD"] = synteyes_array[0]
    eye["LT"] = synteyes_array[1]
    eye["AxialLength"] = synteyes_array[2]
    eye["RT"] = 0.2
    eye["VD"] = (
        synteyes_array[2]
        - synteyes_array[0]
        - synteyes_array[1]
        - synteyes_array[96]
        - eye["RT"]
    )
    eye["Rla"] = synteyes_array[3]
    eye["Rlp"] = synteyes_array[4]
    eye["Qla"] = -3.1316  # fixed, Navarro eye model value
    eye["Qlp"] = -1.0  # fixed, Navarro eye model value
    eye["Pupil"] = 5.0  # mm, fixed
    eye["nc"] = 1.376
    eye["na"] = 1.336
    eye["nv"] = 1.336
    num5 = synteyes_array[5]

    # Lens index solved so the eye's optics reproduce the sampled biometry.
    eye["nl"] = (
        1000
        * (eye["nv"] * (eye["LT"] - eye["Rla"]) + eye["na"] * (eye["LT"] + eye["Rlp"]))
        + num5 * eye["Rla"] * eye["Rlp"]
        - np.sqrt(
            -4e6
            * eye["na"]
            * eye["nv"]
            * eye["LT"]
            * (eye["LT"] - eye["Rla"] + eye["Rlp"])
            + (
                1000 * eye["nv"] * (-eye["LT"] + eye["Rla"])
                + 1000 * eye["na"] * (-eye["LT"] - eye["Rlp"])
                - num5 * eye["Rla"] * eye["Rlp"]
            )
            ** 2
        )
    ) / (2000 * (eye["LT"] - eye["Rla"] + eye["Rlp"]))

    cor_idx = zernike_index(8)
    cor_ant, cor_post = synteyes_array[6:51], synteyes_array[51:96]
    for (n, m), value in zip(cor_idx, cor_ant):
        eye[f"CorAntZ({n:.0f},{m:.0f})"] = value
    for (n, m), value in zip(cor_idx, cor_post):
        eye[f"CorPostZ({n:.0f},{m:.0f})"] = value

    return eye


def _add_retina_curvature(eyes: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Sample retina ellipsoid radii (rx, ry, rz) conditioned on each eye's axial length."""
    axial_lengths = eyes["AxialLength"] - eyes["RT"]

    rx, ry, rz = [], [], []
    for axial_length in axial_lengths:
        cond_mean, cond_cov = conditional_gaussian(
            MU_AL_RADII, COV_AL_RADII, [0], np.array([axial_length])
        )
        sample = rng.multivariate_normal(cond_mean, cond_cov)
        rx.append(sample[0])
        ry.append(sample[1])
        rz.append(sample[2])

    eyes = eyes.copy()
    eyes["retina_rx"] = rx
    eyes["retina_ry"] = ry
    eyes["retina_rz"] = rz
    return eyes


def generate_synteyes(n: int, seed: int | None = None) -> pd.DataFrame:
    """Generate `n` 3D synthetic eyes (SyntEyes plus the van Dam retina extension)."""
    rng = np.random.default_rng(seed)
    modeldata = _load_modeldata()

    eigencorneas = _sample_gmm_eigencornea(modeldata, n, rng)
    eyes = pd.DataFrame(
        [_convert_single_eigencornea(e, modeldata) for e in eigencorneas]
    )
    return _add_retina_curvature(eyes, rng)
