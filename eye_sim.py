"""Ray-tracing simulations for "Optical origin of wide-field angular linearity in the human eye".

Tests the linearity of the field-angle to retinal-angle mapping about the second nodal point N'
(Haasjes et al., Med Phys 2025, doi 10.1002/mp.17576) in schematic and synthetic eyes.

Pipeline: spec dict -> `build_eye_model` -> `trace_eye` -> `compute_metrics`.
Experiments:
  navarro          single-eye checks on the Navarro model
  synteyes         SyntEyes population, phakic plus two pseudophakic IOL designs on the same eyes
  parameter_sweep  Latin-hypercube sweep over 10 geometry parameters

Section A: specs and eye models. B: tracing. C: metrics. D: workers, CSV output, CLI.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

# Set before numpy is imported, otherwise each worker process starts its own BLAS thread pool.
for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

import concurrent.futures

import numpy as np
import pandas as pd
import synteyes
import visisipy
from scipy import stats
from scipy.optimize import brentq
from scipy.stats import qmc
from visisipy.models.materials import (
    EyeMaterials,
    MaterialModel,
    NavarroMaterials458,
    NavarroMaterials543,
    NavarroMaterials589,
    NavarroMaterials633,
)

# Conventions
#
# Mapping error: alpha - theta (retinal angle about the reference point minus field angle);
#   negative = retinal angle smaller than field angle. `error_mm_40` is the unsigned Euclidean
#   distance at 40 deg.
# Retinal shape: a = transverse semi-axis (retina_rx / retina_x_radius), b = axial semi-axis
#   (retina_rz / retina_z_radius), both positive. Vertex radius retina_R = a**2/b (> 0), conic
#   retina_Q = a**2/b**2 - 1 (== retina_asphericity). visisipy takes a signed radius, so
#   geometry.retina.radius = -retina_R for a posterior-curving retina. Checked in
#   run_retina_shape_canary.
# Relative peripheral refraction: rpr_40 = periph_SE_40 - central_SE at ERROR_MM_REFERENCE_ANGLE,
#   pupil diameter PUPIL_DIAMETER_MM.


RunMode = Literal["dev", "full"]

# Refractive indices: SyntEyes cornea/aqueous/vitreous for every eye; lens index from
# spec["materials"]["nl"] when present, else Navarro (see `_spec_materials`); IOLs use spec["iol_index"].

# Must match visisipy's backend wavelength (default 0.543 um); selects the Navarro materials class.
NAVARRO_WAVELENGTH_NM = 543
_NAVARRO_MATERIALS_BY_WAVELENGTH_NM: dict[int, type[EyeMaterials]] = {
    458: NavarroMaterials458,
    543: NavarroMaterials543,
    589: NavarroMaterials589,
    633: NavarroMaterials633,
}

# Field-angle grid for all experiments.
ECCENTRICITIES = np.arange(0.0, 81.0, 5.0)
REFERENCE_POINTS = ("pupil", "second_nodal_point", "retina_center", "corneaC")

# Aperture stop diameter for all experiments. visisipy takes the semi-diameter.
PUPIL_DIAMETER_MM = 5.0
PUPIL_SEMI_DIAMETER_MM = PUPIL_DIAMETER_MM / 2

# Reference-vertex scan grid
Z_REF_MIN_MM = 2.0
Z_REF_MAX_MM = 13.0
Z_REF_STEP_MM = 0.05

# Reference angles in the summary CSV: error_mm_40 and rpr_40 at 40 deg; T_ang_20/T_piv_20 at 20 deg;
# T_ret_50 at 50 deg; linearity and unit-slope fits over <= 60 deg.
ERROR_MM_REFERENCE_ANGLE = 40.0
ANGLE_PIVOT_TERM_ANGLE = 20.0
RETINA_TERM_ANGLE = 50.0
LINEARITY_MAX_ANGLE = 60.0

OUTPUT_DIR = Path(__file__).resolve().parent / "results"

###########################################################
# Section A: geometries. Spec dicts -> visisipy eye models.

LensState = Literal["phakic", "aphakic", "pseudophakic"]


class SpecMaterials(TypedDict):
    """Per-eye refractive indices from SyntEyes. Only `nl` is used (see `_spec_materials`)."""

    nc: float
    na: float
    nl: float
    nv: float


class _EyeSpecRequired(TypedDict):
    """Keys present for every lens_state. Separate class so that `EyeSpec` can mark the rest optional"""

    lens_state: LensState
    cornea_front_radius: float
    cornea_front_asphericity: float
    cornea_back_radius: float
    acd: float  # anterior chamber depth; for a pseudophakic eye the IOL's ELP
    retina_radius: (
        float  # signed, -retina_R for a posterior-curving retina (see Conventions)
    )
    retina_asphericity: float


class EyeSpec(_EyeSpecRequired, total=False):
    """One eye as a plain dict: the geometry `build_eye_model` reads plus pipeline bookkeeping.

    All keys here are optional; `build_eye_model` lists which are required per lens_state.
    """

    # Cornea/pupil geometry; each falls back to NAVARRO_DEFAULTS.
    cornea_thickness: float
    cornea_back_asphericity: float
    pupil_semi_diameter: float
    pupil_lens_distance: float  # subtracted from acd, so axial_length is unchanged
    cornea_back_front_ratio: float  # navarro_spec derives cornea_back_radius from this

    # Crystalline lens (phakic) / IOL (pseudophakic). An aphakic eye needs neither radius.
    lens_front_radius: float
    lens_back_radius: float
    lens_front_asphericity: float
    lens_back_asphericity: float
    lens_thickness: float  # phakic/aphakic; pseudophakic eyes use iol_thickness
    vitreous_depth: (
        float
    )

    # Pseudophakic IOL. iol_asphericity applies to both surfaces unless the per-surface keys are given.
    iol_asphericity: float
    iol_front_asphericity: float
    iol_back_asphericity: float
    iol_thickness: float
    iol_index: float
    axial_length_total: float  # held fixed by the IOL solver while thickness(x) varies

    materials: SpecMaterials

    # Pipeline bookkeeping; removed from the CSV rows via _SPEC_INTERNAL_KEYS.
    eye_id: int
    design: str
    eccentricities: np.ndarray
    compute_defocus: bool
    compute_chief_ray: bool
    needs_iol_solve: bool
    error: str  # set by a generator on failure; `process_eye` passes it through

    # Carried to the summary CSV; not read by `build_eye_model`.
    retina_R: float
    retina_Q: float
    retina_rx: float
    retina_rz: float
    retina_x_radius: float
    retina_z_radius: float
    axial_length_sampled: float
    elp_mm: float


# The Navarro-structured eye, used (with overrides) by every experiment.

NAVARRO_DEFAULTS: dict[str, float] = dict(
    cornea_front_radius=7.72,
    cornea_front_asphericity=-0.26,
    cornea_back_front_ratio=6.50 / 7.72,
    cornea_back_asphericity=0.0,
    cornea_thickness=0.55,
    acd=3.05,
    pupil_semi_diameter=PUPIL_SEMI_DIAMETER_MM,
    lens_front_radius=10.2,
    lens_front_asphericity=-3.1316,
    lens_thickness=4.0,
    lens_back_radius=-6.0,
    lens_back_asphericity=-1.0,
    vitreous_depth=16.3203,
    retina_radius=-12.0,
    retina_asphericity=0.0,
)


def navarro_spec(**overrides) -> EyeSpec:
    """Phakic spec from the Navarro defaults with optional overrides.

    cornea_back_radius = cornea_front_radius * cornea_back_front_ratio, so overriding the front radius
    scales the back radius with it.
    """
    spec: dict[str, Any] = {"lens_state": "phakic", **NAVARRO_DEFAULTS}
    spec.update(overrides)
    spec["cornea_back_radius"] = (
        spec["cornea_front_radius"] * spec["cornea_back_front_ratio"]
    )
    return cast(EyeSpec, spec)


def _default_materials() -> EyeMaterials:
    """Navarro cornea/aqueous/lens/vitreous indices at NAVARRO_WAVELENGTH_NM."""
    return _NAVARRO_MATERIALS_BY_WAVELENGTH_NM[NAVARRO_WAVELENGTH_NM]()


# SyntEyes cornea/aqueous/vitreous indices (constant across eyes; only nl is solved per eye).
SYNTEYES_CORNEA_INDEX = 1.376
SYNTEYES_AQUEOUS_INDEX = 1.336
SYNTEYES_VITREOUS_INDEX = 1.336


def _spec_materials(spec: EyeSpec) -> EyeMaterials:
    """SyntEyes cornea/aqueous/vitreous indices; lens index from spec["materials"]["nl"] when present,
    else the Navarro value. No dispersion is modelled (abbe_number = partial_dispersion = 0).
    """
    custom = spec.get("materials")
    lens_index = (
        custom["nl"]
        if custom is not None
        else _default_materials().lens.refractive_index
    )
    return EyeMaterials(
        cornea=MaterialModel(
            SYNTEYES_CORNEA_INDEX, abbe_number=0, partial_dispersion=0
        ),
        aqueous=MaterialModel(
            SYNTEYES_AQUEOUS_INDEX, abbe_number=0, partial_dispersion=0
        ),
        lens=MaterialModel(lens_index, abbe_number=0, partial_dispersion=0),
        vitreous=MaterialModel(
            SYNTEYES_VITREOUS_INDEX, abbe_number=0, partial_dispersion=0
        ),
    )


def build_eye_model(spec: EyeSpec) -> visisipy.EyeModel:
    """Build a visisipy EyeModel from an EyeSpec.

    Required keys beyond those in `_EyeSpecRequired`:
      phakic:       lens_front_radius, lens_back_radius, lens_thickness, vitreous_depth
                    (lens_*_asphericity optional, Navarro defaults)
      aphakic:      lens_thickness, vitreous_depth. Lens surfaces are flat dummies and lens and vitreous
                    take the aqueous index. Flat surfaces rather than NoSurface() so that plot_eye
                    still works (Fig. 1).
      pseudophakic: lens_front_radius, lens_back_radius, vitreous_depth; optional iol_asphericity
                    (both surfaces) or iol_front_asphericity / iol_back_asphericity, iol_index
                    (default 1.46), iol_thickness (default 1.0 mm).

    pupil_lens_distance (default 0) places the stop that far in front of the lens/IOL front surface,
    taken out of acd so the axial length is unchanged; needed for a concave-anterior IOL.
    """
    lens_state: LensState = spec["lens_state"]

    geometry = visisipy.NavarroGeometry()
    geometry.cornea_front.radius = spec["cornea_front_radius"]
    geometry.cornea_front.asphericity = spec["cornea_front_asphericity"]
    geometry.cornea_front.thickness = spec.get(
        "cornea_thickness", NAVARRO_DEFAULTS["cornea_thickness"]
    )
    geometry.cornea_back.radius = spec["cornea_back_radius"]
    geometry.cornea_back.asphericity = spec.get(
        "cornea_back_asphericity", NAVARRO_DEFAULTS["cornea_back_asphericity"]
    )
    pupil_lens_distance = spec.get("pupil_lens_distance", 0.0)
    geometry.cornea_back.thickness = spec["acd"] - pupil_lens_distance
    geometry.pupil.thickness = pupil_lens_distance
    geometry.pupil.semi_diameter = spec.get(
        "pupil_semi_diameter", NAVARRO_DEFAULTS["pupil_semi_diameter"]
    )

    materials = _spec_materials(spec)

    if lens_state == "phakic":
        geometry.lens_front.radius = spec["lens_front_radius"]
        geometry.lens_front.asphericity = spec.get(
            "lens_front_asphericity", NAVARRO_DEFAULTS["lens_front_asphericity"]
        )
        geometry.lens_front.thickness = spec["lens_thickness"]
        geometry.lens_back.radius = spec["lens_back_radius"]
        geometry.lens_back.asphericity = spec.get(
            "lens_back_asphericity", NAVARRO_DEFAULTS["lens_back_asphericity"]
        )
        geometry.lens_back.thickness = spec["vitreous_depth"]
    elif lens_state == "aphakic":
        geometry.lens_front.radius = float("inf")
        geometry.lens_front.asphericity = 0.0
        geometry.lens_front.thickness = spec["lens_thickness"]
        geometry.lens_back.radius = float("inf")
        geometry.lens_back.asphericity = 0.0
        geometry.lens_back.thickness = spec["vitreous_depth"]
        materials.lens = materials.aqueous
        materials.vitreous = materials.aqueous
    elif lens_state == "pseudophakic":
        default_conic = spec.get("iol_asphericity", 0.0)
        geometry.lens_front.radius = spec["lens_front_radius"]
        geometry.lens_front.asphericity = spec.get(
            "iol_front_asphericity", default_conic
        )
        geometry.lens_front.thickness = spec.get("iol_thickness", 1.0)
        geometry.lens_back.radius = spec["lens_back_radius"]
        geometry.lens_back.asphericity = spec.get("iol_back_asphericity", default_conic)
        geometry.lens_back.thickness = spec["vitreous_depth"]
        materials.lens = MaterialModel(
            spec.get("iol_index", 1.46), abbe_number=0, partial_dispersion=0
        )
    else:
        raise ValueError(f"Unknown lens_state: {lens_state!r}")

    geometry.retina.radius = spec["retina_radius"]
    geometry.retina.asphericity = spec["retina_asphericity"]

    return visisipy.EyeModel(geometry=geometry, materials=materials)


###################################################
# Navarro single-eye checks ("navarro" experiment).
NAVARRO_PIVOT_ANGLES = (15.0, 30.0, 45.0, 60.0)
NAVARRO_ASPHERICITY_VALUES = np.linspace(0.3, -0.82, 9)


def _build_single_surface_model(
    radius: float = 7.8, distance_to_retina: float = 20.0, index: float = 1.336
) -> visisipy.EyeModel:
    """Single refracting surface with the aperture stop at its centre of curvature (Fig. 1C4).

    Chief rays through the stop are undeviated, so alpha == theta exactly; used as a check on the
    tracer. All media have the same index so the dummy surfaces cannot refract.
    """
    from visisipy.models.geometry import NoSurface

    geometry = visisipy.NavarroGeometry()
    geometry.cornea_front.radius = radius
    geometry.cornea_front.asphericity = 0.0
    geometry.cornea_front.thickness = radius  # stop at the centre of curvature
    geometry.cornea_back = NoSurface()
    geometry.pupil.thickness = distance_to_retina
    geometry.lens_front = NoSurface()
    geometry.lens_back = NoSurface()
    geometry.retina.radius = -10.0 * radius
    geometry.retina.asphericity = 0.0

    materials = EyeMaterials(
        cornea=MaterialModel(index, abbe_number=0, partial_dispersion=0),
        aqueous=MaterialModel(index, abbe_number=0, partial_dispersion=0),
        lens=MaterialModel(index, abbe_number=0, partial_dispersion=0),
        vitreous=MaterialModel(index, abbe_number=0, partial_dispersion=0),
    )
    return visisipy.EyeModel(geometry=geometry, materials=materials)


def _navarro_demo_spec(**overrides) -> EyeSpec:
    """Navarro spec with cornea_back_asphericity = 0.0 (the -0.30 default is used elsewhere); used by
    the navarro_* checks only.
    """
    return navarro_spec(cornea_back_asphericity=0.0, **overrides)


###################################################################
# SyntEyes phakic baseline ("synteyes" experiment, design="phakic").

CORNEA_ZERNIKE_RADIUS_MM = synteyes.CORNEA_ZERNIKE_DIAMETER_MM / 2

# Population mean/SD of anterior corneal asphericity over a 6 mm zone, Yazdani et al. 2016
# (doi 10.1007/s10792-016-0334-0). Used instead of the Zernike-derived value (see `_zernike_to_radius`).
CORNEA_FRONT_ASPHERICITY_MEAN = -0.24
CORNEA_FRONT_ASPHERICITY_SD = 0.10


def _zernike_to_radius_asphericity(
    c2: float, c4: float, rho_0: float
) -> tuple[float, float]:
    """Closed-form (radius, asphericity) from standard double-index schema coefficients (mm)."""
    denom = np.sqrt(3) * c2 - 3 * np.sqrt(5) * c4
    radius = rho_0**2 / (4 * denom)
    asphericity = 3 * np.sqrt(5) * c4 * rho_0**2 / (4 * denom**3) - 1
    return radius, asphericity


def _zernike_to_radius(c2: float, c4: float, rho_0: float) -> float:
    """Closed-form radius from standard double-index Zernike coefficients (mm).

    The asphericity from `_zernike_to_radius_asphericity` depends on c4 alone and is too poorly
    conditioned to use per eye; the front asphericity is sampled from a clinical distribution instead
    (`_sample_cornea_front_asphericity`).
    """
    denom = np.sqrt(3) * c2 - 3 * np.sqrt(5) * c4
    return rho_0**2 / (4 * denom)


def _retina_R_Q(a: float, b: float) -> tuple[float, float]:
    """Vertex radius R = a**2/b and conic constant Q = a**2/b**2 - 1 of the retinal ellipsoid.

    a = transverse semi-axis (retina_rx), b = axial semi-axis (retina_rz), both positive. Consistent
    with visisipy's `StandardSurface.ellipsoid_radii` (axial = R/(Q+1), radial = R/sqrt(Q+1)) and
    verified against ray-traced sag in `run_retina_shape_canary`. See Conventions for the sign of the
    radius passed to visisipy.
    """
    return a**2 / b, a**2 / b**2 - 1


# Separate generator so that this draw does not disturb the SyntEyes RNG stream.
_CORNEA_ASPHERICITY_RNG = np.random.default_rng()


def _sample_cornea_front_asphericity(n_eyes: int, seed: int) -> np.ndarray:
    """Front corneal asphericity ~ N(CORNEA_FRONT_ASPHERICITY_MEAN, CORNEA_FRONT_ASPHERICITY_SD).

    Re-seeded on every call so that the phakic and pseudophakic cohorts (same n_eyes, seed) get
    identical values per eye_id.
    """
    _CORNEA_ASPHERICITY_RNG.bit_generator.state = np.random.PCG64(seed).state
    return _CORNEA_ASPHERICITY_RNG.normal(
        CORNEA_FRONT_ASPHERICITY_MEAN, CORNEA_FRONT_ASPHERICITY_SD, size=n_eyes
    )


def generate_synteyes_specs(n_eyes: int, seed: int) -> list[EyeSpec]:
    """SyntEyes phakic cohort.

    Front corneal radius from the SyntEyes Zernike coefficients, front asphericity from
    `_sample_cornea_front_asphericity`; back cornea radius and asphericity both Zernike-derived.
    """
    eyes = synteyes.generate_synteyes(n_eyes, seed=seed)
    front_asphericities = _sample_cornea_front_asphericity(len(eyes), seed)

    specs: list[EyeSpec] = []
    for eye_id, row in eyes.iterrows():
        cornea_front_r = _zernike_to_radius(
            row["CorAntZ(2,0)"], row["CorAntZ(4,0)"], CORNEA_ZERNIKE_RADIUS_MM
        )
        cornea_front_q = front_asphericities[eye_id]
        cornea_back_r, cornea_back_q = _zernike_to_radius_asphericity(
            row["CorPostZ(2,0)"], row["CorPostZ(4,0)"], CORNEA_ZERNIKE_RADIUS_MM
        )

        rx, rz = row["retina_rx"], -row["retina_rz"]
        retina_R, retina_Q = _retina_R_Q(row["retina_rx"], row["retina_rz"])
        specs.append(
            {
                "eye_id": int(eye_id),
                "design": "phakic",
                "lens_state": "phakic",
                "compute_defocus": True,
                "compute_chief_ray": True,
                "cornea_front_radius": cornea_front_r,
                "cornea_front_asphericity": cornea_front_q,
                "cornea_thickness": row["CCT"],
                "cornea_back_radius": cornea_back_r,
                "cornea_back_asphericity": cornea_back_q,
                "acd": row["ACD"],
                "pupil_semi_diameter": PUPIL_SEMI_DIAMETER_MM,
                "lens_front_radius": row["Rla"],
                "lens_front_asphericity": row["Qla"],
                "lens_thickness": row["LT"],
                "lens_back_radius": row["Rlp"],
                "lens_back_asphericity": row["Qlp"],
                "vitreous_depth": row["AxialLength"]
                - row["CCT"]
                - row["ACD"]
                - row["LT"],
                "retina_radius": rx**2 / rz,
                "retina_asphericity": (rx**2 / rz**2) - 1,
                "retina_rx": row["retina_rx"],
                "retina_rz": row["retina_rz"],
                "retina_R": retina_R,
                "retina_Q": retina_Q,
                "axial_length_sampled": row["AxialLength"],
                "materials": {
                    "nc": row["nc"],
                    "na": row["na"],
                    "nl": row["nl"],
                    "nv": row["nv"],
                },
            }
        )
    return specs


def _refraction_M(spec: EyeSpec) -> float:
    model = build_eye_model(spec)
    return float(visisipy.analysis.refraction(model).M)


#############################################################
# SyntEyes pseudophakic cohort ("synteyes" experiment, design=<name>).
#
# Same eyes as the phakic cohort; one row per eye per IOL design (meniscus at ELP 4.0 mm, aspheric
# biconvex at 4.5 mm). Each design is a one-parameter shape family (radii, thickness and conics as
# polynomials in a design point x). Here x is solved per eye for on-axis emmetropia.

IOL_DESIGNS_PATH = Path(__file__).resolve().parent / "iol_designs.json"


def _load_iol_designs(path: Path = IOL_DESIGNS_PATH) -> dict[str, dict]:
    """Returns {} if the file is missing;
    the pseudophakic cohort is then empty.
    """
    if not path.exists():
        warnings.warn(
            f"{path} not found; IOL_DESIGNS is empty until it's generated.",
            stacklevel=2,
        )
        return {}
    with open(path) as f:
        return json.load(f)


IOL_DESIGNS: dict[str, dict] = _load_iol_designs()

PSEUDOPHAKIC_DESIGNS: dict[str, dict] = {
    name: dict(
        elp_mm=design["elp_mm"],
        pupil_lens_distance=design["pupil_lens_distance_mm"],
        iol_index=design["iol_index"],
        compute_defocus=True,
    )
    for name, design in IOL_DESIGNS.items()
}


def _family_poly(coeffs: tuple[float, ...], x: float) -> float:
    """c0 + c1*x + c2*x**2 + ..."""
    return float(sum(c * x**i for i, c in enumerate(coeffs)))


def _iol_design_shape(design: dict, x: float) -> dict[str, float]:
    """Front/back radius, thickness and front/back conic of an IOL design at design point x."""
    coeffs = design["coefficients"]
    return {
        k: _family_poly(tuple(coeffs[k]), x)
        for k in (
            "front_radius",
            "back_radius",
            "thickness",
            "front_conic",
            "back_conic",
        )
    }


def _solve_iol_design(base_spec: EyeSpec, design: dict) -> tuple[EyeSpec, float]:
    """Solve the design point x of an IOL family for on-axis emmetropia (brentq on refraction().M).

    vitreous_depth is recomputed from axial_length_total at each trial x because the IOL thickness
    varies with x. Returns (spec, x); raises if design["x_bracket"] contains no root (logged to
    *_failed.csv by `process_eye`).
    """
    bracket = tuple(design["x_bracket"])

    def _spec_at(x: float) -> EyeSpec:
        shape = _iol_design_shape(design, x)
        return cast(
            EyeSpec,
            {
                **base_spec,
                "lens_front_radius": shape["front_radius"],
                "lens_back_radius": shape["back_radius"],
                "iol_front_asphericity": shape["front_conic"],
                "iol_back_asphericity": shape["back_conic"],
                "iol_thickness": shape["thickness"],
                "vitreous_depth": base_spec["axial_length_total"]
                - base_spec["cornea_thickness"]
                - base_spec["acd"]
                - shape["thickness"],
            },
        )

    x = brentq(lambda x: _refraction_M(_spec_at(x)), *bracket, xtol=1e-5)
    return _spec_at(x), x


def generate_pseudophakic_specs(mode: RunMode) -> list[EyeSpec]:
    """SyntEyes pseudophakic cohort: the phakic cohort's eyes with the lens replaced by an IOL.

    One spec per eye per design in PSEUDOPHAKIC_DESIGNS, with acd = that design's ELP and the
    design's own pupil_lens_distance and iol_index. The IOL shape is solved in the worker
    (`_process_pseudophakic`).
    """
    n = (
        EXPERIMENT_SIZES_FULL["synteyes"]
        if mode == "full"
        else EXPERIMENT_SIZES_DEV["synteyes"]
    )
    eyes = synteyes.generate_synteyes(n, seed=SEEDS["synteyes"])
    front_asphericities = _sample_cornea_front_asphericity(len(eyes), SEEDS["synteyes"])

    specs: list[EyeSpec] = []
    for eye_id, row in eyes.iterrows():
        cornea_front_r = _zernike_to_radius(
            row["CorAntZ(2,0)"], row["CorAntZ(4,0)"], CORNEA_ZERNIKE_RADIUS_MM
        )
        cornea_front_q = front_asphericities[eye_id]
        cornea_back_r, cornea_back_q = _zernike_to_radius_asphericity(
            row["CorPostZ(2,0)"], row["CorPostZ(4,0)"], CORNEA_ZERNIKE_RADIUS_MM
        )
        rx, rz = row["retina_rx"], -row["retina_rz"]
        retina_R, retina_Q = _retina_R_Q(row["retina_rx"], row["retina_rz"])

        base_fields: dict[str, Any] = {
            "lens_state": "pseudophakic",
            "cornea_front_radius": cornea_front_r,
            "cornea_front_asphericity": cornea_front_q,
            "cornea_thickness": row["CCT"],
            "cornea_back_radius": cornea_back_r,
            "cornea_back_asphericity": cornea_back_q,
            "pupil_semi_diameter": PUPIL_SEMI_DIAMETER_MM,
            "retina_radius": rx**2 / rz,
            "retina_asphericity": (rx**2 / rz**2) - 1,
            "retina_R": retina_R,
            "retina_Q": retina_Q,
            "axial_length_total": float(row["AxialLength"]),
        }

        for design_name, design in PSEUDOPHAKIC_DESIGNS.items():
            elp = design["elp_mm"]
            iol_index = design["iol_index"]
            specs.append(
                cast(
                    EyeSpec,
                    {
                        "needs_iol_solve": True,
                        "eye_id": int(eye_id),
                        "design": design_name,
                        "elp_mm": elp,
                        "acd": elp,
                        "axial_length_sampled": row["AxialLength"],
                        "retina_rx": row["retina_rx"],
                        "retina_rz": row["retina_rz"],
                        "pupil_lens_distance": design["pupil_lens_distance"],
                        "iol_index": iol_index,
                        "materials": {
                            "nc": row["nc"],
                            "na": row["na"],
                            "nl": iol_index,
                            "nv": row["nv"],
                        },
                        "compute_defocus": design["compute_defocus"],
                        **base_fields,
                    },
                )
            )
    return specs


########################################################
# Parameter sweep ("parameter_sweep" experiment).

# The retinal semi-axes are sampled; retina_R and retina_Q are derived from them (`_retina_R_Q`).
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "cornea_front_radius": (7.00, 8.60),
    "cornea_front_asphericity": (-0.8, 0.1),
    "cornea_back_front_ratio": (0.70, 0.95),
    "acd": (1.8, 4.4),
    "lens_front_radius": (6.0, 14.0),
    "lens_thickness": (3, 6),
    "lens_back_radius": (-10.0, -4.0),
    "vitreous_depth": (14.00, 22.00),
    "retina_x_radius": (10.4, 14.6),
    "retina_z_radius": (9.1, 12.4),
}
PARAM_NAMES = list(PARAM_RANGES.keys())
SWEEP_FIXED_CORNEA_BACK_ASPHERICITY = -0.30


def generate_parameter_sweep_specs(n_samples: int, seed: int) -> list[EyeSpec]:
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=seed)
    unit_samples = sampler.random(n=n_samples)
    lower = [PARAM_RANGES[name][0] for name in PARAM_NAMES]
    upper = [PARAM_RANGES[name][1] for name in PARAM_NAMES]
    scaled = qmc.scale(unit_samples, lower, upper)
    samples = pd.DataFrame(scaled, columns=PARAM_NAMES)

    specs: list[EyeSpec] = []
    for eye_id, row in samples.iterrows():
        x_radius, z_radius = row["retina_x_radius"], row["retina_z_radius"]
        retina_R, retina_Q = _retina_R_Q(x_radius, z_radius)
        specs.append(
            {
                "eye_id": int(eye_id),
                "lens_state": "phakic",
                "compute_defocus": False,
                "compute_chief_ray": True,
                "cornea_front_radius": row["cornea_front_radius"],
                "cornea_front_asphericity": row["cornea_front_asphericity"],
                "cornea_back_radius": row["cornea_front_radius"]
                * row["cornea_back_front_ratio"],
                "cornea_back_asphericity": SWEEP_FIXED_CORNEA_BACK_ASPHERICITY,
                "acd": row["acd"],
                "lens_front_radius": row["lens_front_radius"],
                "lens_thickness": row["lens_thickness"],
                "lens_back_radius": row["lens_back_radius"],
                "vitreous_depth": row["vitreous_depth"],
                "retina_radius": -retina_R,  # signed, see Conventions
                "retina_asphericity": retina_Q,
                "retina_x_radius": x_radius,
                "retina_z_radius": z_radius,
                "retina_R": retina_R,
                "retina_Q": retina_Q,
                "cornea_back_front_ratio": row["cornea_back_front_ratio"],
            }
        )
    return specs


########################################################
# Section B: tracing.


def trace_eye(
    model: visisipy.EyeModel,
    eccentricities: np.ndarray = ECCENTRICITIES,
) -> dict[str, Any]:
    """Trace chief rays at `eccentricities` (deg) along the horizontal meridian.

    Returns the landing points (z, h), the angle about each point in REFERENCE_POINTS, the reference
    z-positions, and for the vitreous segment of each ray its angle to the axis `beta_deg` (the
    manuscript's beta) and axis crossing `z_Q`, taken from the last two intersections (lens back
    surface and retina).
    """
    coordinates = [(float(e), 0.0) for e in eccentricities]
    height_col = "x"

    geometry = model.geometry
    raytrace_results = visisipy.analysis.raytrace(model, coordinates=coordinates)
    cardinal = visisipy.analysis.cardinal_points(model)
    axial_length = geometry.axial_length

    z_ref = {
        # The stop lies pupil_lens_distance in front of the lens, hence cornea_back.thickness, not acd.
        "pupil": geometry.cornea_thickness + geometry.cornea_back.thickness,
        "second_nodal_point": cardinal.nodal_points.image + axial_length,
        "retina_center": axial_length + geometry.retina.ellipsoid_radii.z,
        "corneaC": geometry.cornea_front.radius,
    }

    rows = []
    for field, group in raytrace_results.groupby("field"):
        group = group.reset_index(drop=True)
        last = group.iloc[-1]
        ecc = float(field[0])
        is_valid = (
            last["comment"] == "retina"
            and np.isfinite(last["z"])
            and np.isfinite(last[height_col])
        )
        z_val = float(last["z"]) if is_valid else np.nan
        h_val = float(last[height_col]) if is_valid else np.nan

        beta = z_q = z_prev = h_prev = dz = dh = np.nan
        if is_valid:
            g = group[
                np.isfinite(group["z"]) & np.isfinite(group[height_col])
            ].sort_values("z")
            if len(g) >= 2:
                p_ret, p_prev = g.iloc[-1], g.iloc[-2]
                z_prev, h_prev = float(p_prev["z"]), float(p_prev[height_col])
                dz = float(p_ret["z"]) - z_prev
                dh = float(p_ret[height_col]) - h_prev
                if dz != 0:
                    beta = float(np.degrees(np.arctan2(dh, dz)))
                    if dh != 0:
                        z_q = z_val - h_val * dz / dh  # h = 0 crossing of the segment

        rows.append(
            {
                "theta": ecc,
                "z": z_val,
                "h": h_val,
                "valid": is_valid,
                "beta_deg": beta,
                "z_Q": z_q,
                "z_prev": z_prev,
                "h_prev": h_prev,
                "dz": dz,
                "dh": dh,
            }
        )
    long_df = pd.DataFrame(rows).sort_values("theta").reset_index(drop=True)

    theta = long_df["theta"].to_numpy()
    z = long_df["z"].to_numpy()
    h = long_df["h"].to_numpy()
    valid = long_df["valid"].to_numpy()

    angle = {ref: np.degrees(np.arctan2(h, z - z_ref[ref])) for ref in REFERENCE_POINTS}

    return {
        "theta": theta,
        "z": z,
        "h": h,
        "valid": valid,
        "angle": angle,
        "z_ref": z_ref,
        "axial_length": axial_length,
        "retina_radius": geometry.retina.radius,
        "retina_asphericity": geometry.retina.asphericity,
        "beta_deg": long_df["beta_deg"].to_numpy(),
        "z_Q": long_df["z_Q"].to_numpy(),
        "z_prev": long_df["z_prev"].to_numpy(),
        "h_prev": long_df["h_prev"].to_numpy(),
        "dz": long_df["dz"].to_numpy(),
        "dh": long_df["dh"].to_numpy(),
    }


_SPEC_INTERNAL_KEYS = {
    "materials",
    "eccentricities",
    "compute_defocus",
    "needs_iol_solve",
    "compute_chief_ray",
}
_LONG_ROW_SPEC_KEYS = ("eye_id", "design")


def process_eye(
    spec: EyeSpec,
) -> tuple[list[dict], dict | None, list[dict], dict | None]:
    """Build, trace and compute metrics for one eye spec (worker function).

    Returns (long_rows, summary_row, chief_ray_rows, failure). On failure the first three are empty/None
    and `failure` holds the spec plus the error message. Pseudophakic specs (needs_iol_solve) are
    solved here so that the solve runs in parallel.
    """
    if "error" in spec:
        return (
            [],
            None,
            [],
            {k: v for k, v in spec.items() if k not in _SPEC_INTERNAL_KEYS},
        )

    try:
        if spec.get("needs_iol_solve"):
            trace, metrics = _process_pseudophakic(spec)
        else:
            eccentricities = spec.get("eccentricities", ECCENTRICITIES)
            compute_defocus = spec.get("compute_defocus", False)
            model = build_eye_model(spec)
            trace = trace_eye(model, eccentricities)
            metrics = compute_metrics(
                trace, model=model, compute_defocus=compute_defocus
            )
    except Exception as exc:  # noqa: BLE001 - record the failure and continue
        failure: dict[str, Any] = {
            k: v for k, v in spec.items() if k not in _SPEC_INTERNAL_KEYS
        }
        failure["error"] = str(exc)
        return [], None, [], failure

    return _assemble_eye_result(spec, trace, metrics)


def _chief_ray_summary_terms(decomp_df: pd.DataFrame) -> dict:
    """T_ang and T_piv at ANGLE_PIVOT_TERM_ANGLE and T_ret at RETINA_TERM_ANGLE, for the summary CSV."""

    def _at(angle: float, col: str) -> float:
        row = decomp_df.loc[np.isclose(decomp_df["theta"], angle)]
        return (
            float(row[col].iloc[0])
            if len(row) and bool(row["valid"].iloc[0])
            else float("nan")
        )

    return {
        "T_ang_20": _at(ANGLE_PIVOT_TERM_ANGLE, "T_ang"),
        "T_piv_20": _at(ANGLE_PIVOT_TERM_ANGLE, "T_piv"),
        "T_ret_50": _at(RETINA_TERM_ANGLE, "T_ret"),
    }


def _assemble_eye_result(
    spec: EyeSpec, trace: dict, metrics: dict
) -> tuple[list[dict], dict, list[dict], None]:
    spec_fields = cast(dict[str, Any], spec)
    long_rows = []
    for i in range(len(trace["theta"])):
        row = {k: spec_fields[k] for k in _LONG_ROW_SPEC_KEYS if k in spec_fields}
        row.update(
            {
                "theta": trace["theta"][i],
                "z": trace["z"][i],
                "h": trace["h"][i],
                "valid": bool(trace["valid"][i]),
            }
        )
        row["angle_second_nodal_point"] = trace["angle"]["second_nodal_point"][i]
        long_rows.append(row)

    summary_row = {k: v for k, v in spec.items() if k not in _SPEC_INTERNAL_KEYS}
    summary_row.update(metrics)

    chief_ray_rows = []
    if spec.get("compute_chief_ray", False):
        z_unitslope = metrics.get("z_unitslope", float("nan"))
        decomp_df = decompose_chief_ray(trace, z_unitslope=z_unitslope)
        summary_row.update(_chief_ray_summary_terms(decomp_df))
        for k in _LONG_ROW_SPEC_KEYS:
            if k in spec:
                decomp_df[k] = spec[k]
        chief_ray_rows = decomp_df.to_dict("records")

    return long_rows, summary_row, chief_ray_rows, None


def _init_worker() -> None:
    """The visisipy backend is per-process state; set it in every worker."""
    warnings.filterwarnings("ignore", category=UserWarning)
    visisipy.set_backend("optiland", computation_backend="numpy")


def _process_pseudophakic(spec: EyeSpec) -> tuple[dict, dict]:
    """Solve the IOL shape for on-axis emmetropia, then build, trace and compute metrics."""
    design = IOL_DESIGNS[spec["design"]]
    base_spec = cast(
        EyeSpec, {k: v for k, v in spec.items() if k not in _PSEUDOPHAKIC_NON_BASE_KEYS}
    )
    full_spec, _x = _solve_iol_design(base_spec, design)

    model = build_eye_model(full_spec)
    trace = trace_eye(model, spec.get("eccentricities", ECCENTRICITIES))
    metrics = compute_metrics(
        trace, model=model, compute_defocus=spec.get("compute_defocus", False)
    )
    return trace, metrics


_PSEUDOPHAKIC_NON_BASE_KEYS = {
    "needs_iol_solve",
    "design",
    "elp_mm",
    "axial_length_sampled",
    "retina_rx",
    "retina_rz",
    "eccentricities",
    "compute_defocus",
}

#########################################################################
# Section C: metrics.


def _mapping_metrics(
    theta_deg: np.ndarray, alpha_deg: np.ndarray, max_angle: float = LINEARITY_MAX_ANGLE
) -> dict:
    """Linearity metrics of alpha vs theta over eccentricities <= max_angle: through-origin slope
    (slope_to), OLS slope/intercept/R^2, quadratic coefficient a2, max |alpha - theta| (maxdev_id),
    RMS deviation from the through-origin line (rmsdev_fit), alpha at 60 deg, n_valid.
    """
    valid = np.isfinite(alpha_deg) & (theta_deg <= max_angle)
    n_valid = int(valid.sum())
    keys = [
        "slope_to",
        "slope_ols",
        "intercept_ols",
        "a2",
        "r_squared",
        "maxdev_id",
        "rmsdev_fit",
        "alpha60",
    ]
    if n_valid < 3:
        return {**dict.fromkeys(keys, np.nan), "n_valid": n_valid}

    theta, alpha = theta_deg[valid], alpha_deg[valid]

    slope_to = np.sum(theta * alpha) / np.sum(theta**2)
    ols = stats.linregress(theta, alpha)

    design = np.column_stack([theta, theta**2])
    (a1, a2), *_ = np.linalg.lstsq(design, alpha, rcond=None)

    residuals_to = alpha - slope_to * theta
    maxdev_id = float(np.max(np.abs(alpha - theta)))
    rmsdev_fit = float(np.sqrt(np.mean(residuals_to**2)))
    alpha60 = float(np.interp(60.0, theta, alpha)) if theta.max() >= 60.0 else np.nan

    return {
        "slope_to": slope_to,
        "slope_ols": ols.slope,
        "intercept_ols": ols.intercept,
        "a2": a2,
        "r_squared": ols.rvalue**2,
        "maxdev_id": maxdev_id,
        "rmsdev_fit": rmsdev_fit,
        "alpha60": alpha60,
        "n_valid": n_valid,
    }


def reference_vertex_scan_curve(
    theta_deg: np.ndarray,
    z: np.ndarray,
    h: np.ndarray,
    max_angle: float = LINEARITY_MAX_ANGLE,
) -> dict:
    """Through-origin slope and max |alpha - theta| for every reference point on the z_ref grid."""
    valid = np.isfinite(z) & np.isfinite(h) & (theta_deg <= max_angle)
    theta_v, z_v, h_v = theta_deg[valid], z[valid], np.abs(h[valid])

    z_ref_grid = np.arange(Z_REF_MIN_MM, Z_REF_MAX_MM + Z_REF_STEP_MM, Z_REF_STEP_MM)
    alpha = np.degrees(np.arctan2(h_v[None, :], z_v[None, :] - z_ref_grid[:, None]))

    theta_b = theta_v[None, :]
    slope_to_grid = np.sum(theta_b * alpha, axis=1) / np.sum(theta_b**2)
    maxabsdev_grid = np.max(np.abs(alpha - theta_b), axis=1)

    return {
        "z_ref_grid": z_ref_grid,
        "slope_to_grid": slope_to_grid,
        "maxabsdev_grid": maxabsdev_grid,
    }


def reference_vertex_scan(
    theta_deg: np.ndarray,
    z: np.ndarray,
    h: np.ndarray,
    max_angle: float = LINEARITY_MAX_ANGLE,
) -> dict:
    """Reference point with unit slope (z_unitslope) and with minimal max deviation (z_mindist)."""
    curve = reference_vertex_scan_curve(theta_deg, z, h, max_angle=max_angle)
    z_ref_grid, slope_to_grid, maxabsdev_grid = (
        curve["z_ref_grid"],
        curve["slope_to_grid"],
        curve["maxabsdev_grid"],
    )
    z_unitslope = float(np.interp(1.0, slope_to_grid, z_ref_grid))
    i_mindist = int(np.argmin(maxabsdev_grid))
    return {
        "z_unitslope": z_unitslope,
        "z_mindist": float(z_ref_grid[i_mindist]),
    }


def _predict_landing_point(
    theta_deg: np.ndarray,
    z_ref: float,
    slope: float,
    z_vertex: float,
    radius: float,
    asphericity: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Landing point predicted from alpha = slope * theta about z_ref: line/conic intersection."""
    theta = np.radians(theta_deg)
    tan_alpha = np.tan(slope * theta)
    k = (z_vertex - z_ref) * tan_alpha

    a = tan_alpha**2 + (1 + asphericity)
    b = 2 * k * tan_alpha - 2 * radius
    c = k**2

    with np.errstate(invalid="ignore", divide="ignore"):
        disc = np.sqrt(np.clip(b**2 - 4 * a * c, 0, None))
        root1 = (-b + disc) / (2 * a)
        root2 = (-b - disc) / (2 * a)
    z_prime = np.where(np.abs(root1) <= np.abs(root2), root1, root2)
    z_prime = np.where(theta_deg == 0, 0.0, z_prime)

    h_pred = k + z_prime * tan_alpha
    z_pred = z_vertex + z_prime
    return z_pred, h_pred


def practical_accuracy(
    theta_deg: np.ndarray,
    z: np.ndarray,
    h: np.ndarray,
    z_n2: float,
    axial_length: float,
    retina_radius: float,
    retina_asphericity: float,
) -> dict:
    """Distance between the alpha = theta prediction about N' and the traced landing point."""
    valid = np.isfinite(z) & np.isfinite(h)
    theta_v, z_v, h_v = theta_deg[valid], z[valid], np.abs(h[valid])

    z_pred, h_pred = _predict_landing_point(
        theta_v, z_n2, 1.0, axial_length, retina_radius, retina_asphericity
    )
    error_mm = np.sqrt((z_pred - z_v) ** 2 + (h_pred - h_v) ** 2)

    error_mm_40 = (
        float(np.interp(ERROR_MM_REFERENCE_ANGLE, theta_v, error_mm))
        if theta_v.max() >= ERROR_MM_REFERENCE_ANGLE
        else np.nan
    )
    return {
        "error_mm_40": error_mm_40,
        "error_mm_max": float(np.max(error_mm)) if len(error_mm) else np.nan,
    }


# Refraction is evaluated on axis and at ERROR_MM_REFERENCE_ANGLE.
DEFOCUS_ANGLES_DEG = (0.0, ERROR_MM_REFERENCE_ANGLE)


def _rpr_at(theta_arr: np.ndarray, m_values: np.ndarray, angle: float) -> float:
    """Peripheral minus central spherical equivalent at `angle`, interpolated over the finite values."""
    valid = np.isfinite(m_values)
    if valid.sum() >= 2 and valid[0] and theta_arr.max() >= angle:
        rpr = m_values[valid] - m_values[0]
        return float(np.interp(angle, theta_arr[valid], rpr))
    return np.nan


def peripheral_defocus(
    model: visisipy.EyeModel, angles_deg: np.ndarray = DEFOCUS_ANGLES_DEG
) -> dict:
    """Spherical-equivalent refraction at `angles_deg` (horizontal meridian, pupil PUPIL_DIAMETER_MM).

    Returns central_SE, periph_SE_40 and rpr_40 = periph_SE_40 - central_SE. Only run when
    compute_defocus is set (about 25 ms per angle).
    """
    m_values = []
    for theta in angles_deg:
        coord = (theta, 0.0)
        try:
            m_values.append(
                visisipy.analysis.refraction(model, field_coordinate=coord).M
            )
        except Exception:  # noqa: BLE001
            m_values.append(np.nan)
    m_values = np.array(m_values, dtype=float)
    theta_arr = np.asarray(angles_deg, dtype=float)

    periph_se = {f"periph_SE_{int(a)}": v for a, v in zip(angles_deg, m_values)}
    central_se = float(m_values[0]) if len(m_values) else np.nan
    periph_se_40 = periph_se.get("periph_SE_40", np.nan)
    rpr_40 = _rpr_at(theta_arr, m_values, ERROR_MM_REFERENCE_ANGLE)

    if np.isfinite(periph_se_40) and np.isfinite(central_se) and np.isfinite(rpr_40):
        assert np.isclose(rpr_40, periph_se_40 - central_se, atol=1e-9), (
            f"rpr_40 identity violated: rpr_40={rpr_40!r} != periph_SE_40 - central_SE = {periph_se_40 - central_se!r}"
        )

    return {
        "periph_SE_40": periph_se_40,
        "central_SE": central_se,
        "rpr_40": rpr_40,
    }


def compute_metrics(
    trace: dict, *, model: visisipy.EyeModel, compute_defocus: bool = False
) -> dict:
    """Per-eye metrics: linearity about N', reference-vertex scan, reference z-positions, landing-point
    error, central_SE and (if compute_defocus) periph_SE_40 and rpr_40.
    """
    theta, z, h = trace["theta"], trace["z"], trace["h"]

    metrics: dict[str, Any] = {}
    block = _mapping_metrics(theta, trace["angle"]["second_nodal_point"])
    for key in ("slope_to", "r_squared", "maxdev_id", "n_valid"):
        metrics[f"{key}_second_nodal_point"] = block[key]

    scan = reference_vertex_scan(theta, z, h)
    metrics.update(scan)

    z_n2 = trace["z_ref"]["second_nodal_point"]
    z_corneac = trace["z_ref"]["corneaC"]
    metrics["z_N2"] = z_n2
    metrics["z_corneaC"] = z_corneac
    metrics["z_EP"] = trace["z_ref"]["pupil"]
    metrics["z_retc"] = trace["z_ref"]["retina_center"]
    metrics["sep_N2_corneaC"] = z_n2 - z_corneac
    # M of the manuscript (L - z_N'); not named M to avoid confusion with the refraction M.
    metrics["nodal_to_retina_mm"] = trace["axial_length"] - z_n2

    metrics.update(
        practical_accuracy(
            theta,
            z,
            h,
            z_n2,
            trace["axial_length"],
            trace["retina_radius"],
            trace["retina_asphericity"],
        )
    )

    if compute_defocus:
        metrics.update(peripheral_defocus(model))
    else:
        metrics["central_SE"] = float(visisipy.analysis.refraction(model).M)

    return metrics


# Chief-ray decomposition
#
# Symbols follow the manuscript:
#     theta    field angle of the incoming chief ray
#     beta     angle of the image-space (vitreous) segment of the chief ray to the axis; the real-eye
#              counterpart of the reduced eye's phi (Methods)
#     z_Q      axis crossing of that segment; the counterpart of the reduced eye's z_p (Eq. 4).
#              Equals the exit pupil paraxially; varies with theta for a real ray
#     z_N2     second nodal point z_N'
#     L        axial length
#     (z_P,h_P) landing point of the chief ray on the retina
#     s        = L - z_P, sag of the retina at the landing point (Eq. 3)
#     H, M, delta  = L - z_Q, L - z_N', z_N' - z_Q = H - M (Eq. 4)
#     alpha    retinal angle about N': tan(alpha) = h_P / (z_P - z_N')
#
# The mapping error is split as in Eq. (6), alpha - theta = (beta - theta) + (alpha - beta), with the
# geometric part alpha - beta further split by the pole plane:
#     T_ang       = beta - theta          demagnification by the anterior segment (Eq. 7)
#     T_piv       = alpha_flat - beta     N' rather than z_Q as reference, flat retina (Eq. 8)
#     T_ret       = alpha - alpha_flat    s-dependent remainder of Eq. (5): tan(alpha) -
#                                         tan(alpha_flat) = tan(beta) * delta*s / (M (M - s))
#     total_error = alpha - theta         = T_ang + T_piv + T_ret
# T_geo of Eq. (6) is T_piv + T_ret.
#
# Computed for specs with compute_chief_ray set; `_chief_ray_summary_terms` writes T_ang_20, T_piv_20
# and T_ret_50 to the summary CSV.


def _angle_about(h: float, z: float, z_ref: float) -> float:
    """Signed angle (deg) of the point (z, h) about the axial point z_ref."""
    return float(np.degrees(np.arctan2(h, z - z_ref)))


def decompose_chief_ray(trace: dict, z_unitslope: float = float("nan")) -> pd.DataFrame:
    """Decompose the mapping error alpha - theta into T_ang, T_piv and T_ret per field angle
    (manuscript Eqs. 5-8; symbols as in the comment above).

    Takes a `trace_eye` result. Output columns: z_P, h_P, beta, z_Q, alpha, alpha_flat, s, T_ang, T_piv,
    T_ret, total_error, total_check (angles in deg). z_unitslope is carried through for reference only.
    """
    theta = np.asarray(trace["theta"], float)
    z_P, h_P = np.asarray(trace["z"], float), np.asarray(trace["h"], float)
    beta = np.asarray(trace["beta_deg"], float)
    z_Q = np.asarray(trace["z_Q"], float)
    z_N2 = float(trace["z_ref"]["second_nodal_point"])
    L = float(trace["axial_length"])
    M = L - z_N2  # Eq. (4): nodal point to retinal pole

    rows = []
    for i, th in enumerate(theta):
        if not np.isfinite(z_P[i]) or not np.isfinite(h_P[i]):
            rows.append({"theta": th, "valid": False})
            continue
        alpha = _angle_about(h_P[i], z_P[i], z_N2)
        s = L - z_P[i]  # Eq. (3): retinal sag at the landing point
        if th == 0:  # on axis alpha = beta = 0 and z_Q is undefined
            rows.append(
                {
                    "theta": 0.0,
                    "valid": True,
                    "z_P": z_P[i],
                    "h_P": h_P[i],
                    "beta": 0.0,
                    "z_Q": np.nan,
                    "alpha": 0.0,
                    "alpha_flat": 0.0,
                    "s": s,
                    "T_ang": 0.0,
                    "T_piv": 0.0,
                    "T_ret": 0.0,
                    "total_error": 0.0,
                    "total_check": 0.0,
                }
            )
            continue
        H = L - z_Q[i]  # Eq. (4), with z_Q in the role of z_p
        if not (np.isfinite(beta[i]) and np.isfinite(z_Q[i])) or H <= 0:
            # vitreous segment not resolved, or it crosses the axis behind the pole plane: the flat-retina
            # projection of Eq. (5) with s = 0 has no forward intersection, so pivot and retina cannot be split
            rows.append({"theta": th, "valid": False})
            continue
        # Eq. (5) with s = 0: the same segment continued to the pole plane, read about N'
        alpha_flat = float(np.degrees(np.arctan(np.tan(np.radians(beta[i])) * H / M)))

        T_ang = beta[i] - th  # Eq. (6)/(7)
        T_piv = alpha_flat - beta[i]  # flat-retina part of Eq. (5), Eq. (8)
        T_ret = alpha - alpha_flat  # s-dependent remainder of Eq. (5)
        rows.append(
            {
                "theta": th,
                "valid": True,
                "z_P": z_P[i],
                "h_P": h_P[i],
                "beta": beta[i],
                "z_Q": z_Q[i],
                "alpha": alpha,
                "alpha_flat": alpha_flat,
                "s": s,
                "T_ang": T_ang,
                "T_piv": T_piv,
                "T_ret": T_ret,
                "total_error": alpha - th,
                "total_check": T_ang + T_piv + T_ret,
            }
        )
    df = pd.DataFrame(rows)
    df["z_N2"] = z_N2
    df["z_corneaC"] = trace["z_ref"]["corneaC"]
    df["z_unitslope"] = z_unitslope
    return df


##############################################
# Navarro single-eye checks ("navarro" experiment); expected values are in `run_navarro`.


def navarro_single_surface_check() -> pd.DataFrame:
    model = _build_single_surface_model()
    radius = model.geometry.cornea_front.radius
    trace = trace_eye(model, np.array(NAVARRO_PIVOT_ANGLES))
    alpha_about_centre = np.degrees(np.arctan2(trace["h"], trace["z"] - radius))
    return pd.DataFrame(
        {
            "theta": trace["theta"],
            "alpha_about_centre_of_curvature": alpha_about_centre,
            "abs_dev_deg": np.abs(alpha_about_centre - trace["theta"]),
        }
    )


def navarro_three_pivot_comparison() -> pd.DataFrame:
    model = build_eye_model(_navarro_demo_spec())
    trace = trace_eye(model, np.array(NAVARRO_PIVOT_ANGLES))
    rows = []
    for ref in ("pupil", "second_nodal_point", "retina_center"):
        alpha = trace["angle"][ref]
        metrics = _mapping_metrics(trace["theta"], alpha)
        rows.append(
            {
                "reference": ref,
                "z_ref": trace["z_ref"][ref],
                "slope_to": metrics["slope_to"],
                "a2": metrics["a2"],
            }
        )
    return pd.DataFrame(rows)


def navarro_vertex_scan(
    lens_state: Literal["phakic", "aphakic"] = "phakic",
) -> tuple[pd.DataFrame, dict]:
    spec = _navarro_demo_spec(lens_state=lens_state)
    model = build_eye_model(spec)
    trace = trace_eye(model, ECCENTRICITIES)
    curve = reference_vertex_scan_curve(trace["theta"], trace["z"], trace["h"])
    scan = reference_vertex_scan(trace["theta"], trace["z"], trace["h"])

    n2_metrics = _mapping_metrics(trace["theta"], trace["angle"]["second_nodal_point"])
    summary = {
        "lens_state": lens_state,
        "z_N2": trace["z_ref"]["second_nodal_point"],
        "z_corneaC": trace["z_ref"]["corneaC"],
        "slope_about_N2": n2_metrics["slope_to"],
        **scan,
    }
    curve_df = pd.DataFrame(
        {
            "z_ref": curve["z_ref_grid"],
            "slope_to": curve["slope_to_grid"],
            "maxabsdev": curve["maxabsdev_grid"],
        }
    )
    return curve_df, summary


def navarro_asphericity_sweep() -> pd.DataFrame:
    rows = []
    for k in NAVARRO_ASPHERICITY_VALUES:
        spec = _navarro_demo_spec(cornea_front_asphericity=float(k))
        model = build_eye_model(spec)
        trace = trace_eye(model, ECCENTRICITIES)
        metrics = _mapping_metrics(trace["theta"], trace["angle"]["second_nodal_point"])
        rows.append(
            {
                "cornea_front_asphericity": float(k),
                "z_N2": trace["z_ref"]["second_nodal_point"],
                "slope_to_N2": metrics["slope_to"],
                "maxdev_id_N2": metrics["maxdev_id"],
                "a2_N2": metrics["a2"],
            }
        )
    return pd.DataFrame(rows)


def navarro_chief_ray_decomposition() -> pd.DataFrame:
    """Chief-ray decomposition for the phakic and aphakic Navarro eye (Fig. 3)."""
    decomp_all = []
    for lens_state, label in (
        ("phakic", "navarro_phakic"),
        ("aphakic", "navarro_aphakic"),
    ):
        model = build_eye_model(_navarro_demo_spec(lens_state=lens_state))
        trace = trace_eye(model, ECCENTRICITIES)
        z_unitslope = reference_vertex_scan(trace["theta"], trace["z"], trace["h"])[
            "z_unitslope"
        ]
        decomp = decompose_chief_ray(trace, z_unitslope=z_unitslope)
        decomp["label"] = label
        decomp_all.append(decomp)
    return pd.concat(decomp_all, ignore_index=True)


#############################################
# Section D: workers, CSV output and CLI.


def run_canary(tolerance_deg: float = 1.0) -> float:
    """Check that the Navarro eye gives alpha ~ 59.5 deg at theta = 60 deg about N' (Haasjes et al.
    report 59.9 +/- 0.4 deg for 27 real eyes). Raises on failure.
    """
    model = build_eye_model(navarro_spec())
    trace = trace_eye(model, ECCENTRICITIES)
    idx60 = np.argmin(np.abs(trace["theta"] - 60.0))
    alpha60 = trace["angle"]["second_nodal_point"][idx60]
    print(
        f"Navarro check: theta = 60 deg -> alpha(N') = {alpha60:.2f} deg (expected ~59.5 deg)"
    )
    if not np.isfinite(alpha60):
        raise RuntimeError(
            "Navarro check failed: the 60 deg chief ray did not reach the retina."
        )
    if abs(alpha60 - 59.5) > tolerance_deg:
        raise RuntimeError(
            f"Navarro check failed: {alpha60:.2f} deg differs by more than {tolerance_deg} deg from 59.5 deg."
        )
    print("Navarro check passed.\n")
    return alpha60


def run_retina_shape_canary(n_eyes: int = 5, tolerance: float = 1e-3) -> None:
    """Check the retina_R/retina_Q convention of `_retina_R_Q`.

    Fits R and Q to the traced (h, z) landing points of a few SyntEyes eyes with the conic sag equation
    h**2 = 2*R*u - (1+Q)*u**2 and compares with the analytic values. Run at the start of every
    invocation because a wrong semi-axis assignment would not raise an error.
    """
    print("=== retina shape check: R, Q from sag fit vs analytic ===")
    specs = generate_synteyes_specs(n_eyes, seed=SEEDS["synteyes"])
    max_rel_err = 0.0
    for spec in specs:
        model = build_eye_model(spec)
        trace = trace_eye(model, ECCENTRICITIES)
        theta, z, h = trace["theta"], trace["z"], trace["h"]
        axial_length = trace["axial_length"]
        valid = (theta > 0) & np.isfinite(z) & np.isfinite(h)
        u, hh = z[valid] - axial_length, h[valid]
        design = np.column_stack([u, u**2])
        (coef_2r, coef_neg_1plusq), *_ = np.linalg.lstsq(design, hh**2, rcond=None)
        r_fit, q_fit = coef_2r / 2.0, -coef_neg_1plusq - 1.0

        # The fit returns the signed radius, so compare with -retina_R.
        r_expected, q_expected = -spec["retina_R"], spec["retina_Q"]
        rel_err = max(
            abs(r_fit - r_expected) / abs(r_expected), abs(q_fit - q_expected)
        )
        max_rel_err = max(max_rel_err, rel_err)
        print(
            f"  eye {spec['eye_id']}: R_fit={r_fit:+.4f} (analytic {r_expected:+.4f})  Q_fit={q_fit:+.5f} (analytic {q_expected:+.5f})"
        )

    status = "PASS" if max_rel_err < tolerance else "FAIL"
    print(
        f"  [{status}] max relative error = {max_rel_err:.2e} (tolerance {tolerance:.0e})\n"
    )
    if max_rel_err >= tolerance:
        raise RuntimeError(
            f"Retina shape check failed: max relative error {max_rel_err:.2e} >= {tolerance:.0e}."
        )


def run_navarro(out_dir: Path) -> None:
    """Run the six Navarro checks, write their CSVs and print PASS/WARN against expected values."""
    out_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True

    def check(label: str, ok: bool, detail: str) -> None:
        nonlocal all_ok
        status = "PASS" if ok else "WARN"
        if not ok:
            all_ok = False
        print(f"  [{status}] {label}: {detail}")

    print("=== navarro 1/6: single surface, stop at its own centre of curvature ===")
    single = navarro_single_surface_check()
    single.to_csv(out_dir / "navarro_single_surface_check.csv", index=False)
    max_dev = single["abs_dev_deg"].max()
    check(
        "exactness",
        max_dev < 1e-6,
        f"max |alpha-theta| = {max_dev:.2e} deg (expect ~0, machine precision)",
    )

    print("=== navarro 2/6: three-pivot comparison (pupil / N' / retina-centre) ===")
    three_pivot = navarro_three_pivot_comparison()
    three_pivot.to_csv(out_dir / "navarro_three_pivot.csv", index=False)
    slopes = dict(zip(three_pivot["reference"], three_pivot["slope_to"]))
    check(
        "pupil slope ~0.77",
        0.65 <= slopes["pupil"] <= 0.90,
        f"slope={slopes['pupil']:.3f}",
    )
    check(
        "N' slope ~0.99-1.00",
        0.95 <= slopes["second_nodal_point"] <= 1.05,
        f"slope={slopes['second_nodal_point']:.3f}",
    )
    check(
        "retina-centre slope ~1.31-1.42",
        1.15 <= slopes["retina_center"] <= 1.55,
        f"slope={slopes['retina_center']:.3f}",
    )

    print(
        "=== navarro 3/6: phakic vertex scan (N' ~= corneaC ~= unit-slope ~= min-distortion vertex) ==="
    )
    curve_p, summary_p = navarro_vertex_scan("phakic")
    curve_p.to_csv(out_dir / "navarro_vertex_scan_phakic.csv", index=False)
    cluster_p = [
        summary_p["z_N2"],
        summary_p["z_corneaC"],
        summary_p["z_unitslope"],
        summary_p["z_mindist"],
    ]
    spread_p = max(cluster_p) - min(cluster_p)
    check(
        "phakic cluster tight",
        spread_p < 0.5,
        f"z_N2={summary_p['z_N2']:.3f}  z_corneaC={summary_p['z_corneaC']:.3f}  z_unitslope={summary_p['z_unitslope']:.3f}  z_mindist={summary_p['z_mindist']:.3f}  spread={spread_p:.3f}mm",
    )

    print("=== navarro 4/6: aphakic vertex scan (coincidence breaks) ===")
    curve_a, summary_a = navarro_vertex_scan("aphakic")
    curve_a.to_csv(out_dir / "navarro_vertex_scan_aphakic.csv", index=False)
    separation = summary_a["z_N2"] - summary_a["z_unitslope"]
    check(
        "aphakic separation >0.8mm",
        separation > 0.8,
        f"z_N2={summary_a['z_N2']:.3f}  z_unitslope={summary_a['z_unitslope']:.3f}  separation={separation:.3f}mm",
    )
    check(
        "aphakic slope-about-N' ~1.03-1.12",
        1.03 <= summary_a["slope_about_N2"] <= 1.12,
        f"slope={summary_a['slope_about_N2']:.3f}",
    )

    pd.DataFrame([summary_p, summary_a]).to_csv(
        out_dir / "navarro_vertex_scan_summary.csv", index=False
    )

    print(
        "=== navarro 5/6: corneal-asphericity sweep (N' invariant; residual grows with prolateness) ==="
    )
    asphericity = navarro_asphericity_sweep()
    asphericity.to_csv(out_dir / "navarro_asphericity_sweep.csv", index=False)
    z_n2_spread = asphericity["z_N2"].max() - asphericity["z_N2"].min()
    check(
        "N' paraxially invariant",
        z_n2_spread < 0.1,
        f"z_N2 range = {asphericity['z_N2'].min():.3f}-{asphericity['z_N2'].max():.3f}mm (spread {z_n2_spread:.3f}mm)",
    )
    most_prolate_dev = asphericity.sort_values("cornea_front_asphericity").iloc[0][
        "maxdev_id_N2"
    ]
    least_prolate_dev = asphericity.sort_values("cornea_front_asphericity").iloc[-1][
        "maxdev_id_N2"
    ]
    check(
        "residual grows with prolateness",
        most_prolate_dev > least_prolate_dev,
        f"maxdev_id at k={NAVARRO_ASPHERICITY_VALUES.min():.2f} (most prolate) = {most_prolate_dev:.3f} deg vs k={NAVARRO_ASPHERICITY_VALUES.max():.2f} (least prolate) = {least_prolate_dev:.3f} deg",
    )

    print(
        "=== navarro 6/6: chief-ray error-budget decomposition (Navarro phakic vs aphakic) ==="
    )
    decomp_nav = navarro_chief_ray_decomposition()
    decomp_nav.to_csv(out_dir / "navarro_chief_ray_decomposition.csv", index=False)
    d40p = decomp_nav[
        (decomp_nav["label"] == "navarro_phakic")
        & np.isclose(decomp_nav["theta"], ERROR_MM_REFERENCE_ANGLE)
    ]
    d40a = decomp_nav[
        (decomp_nav["label"] == "navarro_aphakic")
        & np.isclose(decomp_nav["theta"], ERROR_MM_REFERENCE_ANGLE)
    ]
    if len(d40p):
        row = d40p.iloc[0]
        check(
            "decomposition sums to total (phakic@40deg)",
            abs(row["total_check"] - row["total_error"]) < 1e-6,
            f"angle={row['T_ang']:+.3f} pivot={row['T_piv']:+.3f} retina={row['T_ret']:+.3f} total={row['total_error']:+.3f} deg",
        )
    if len(d40p) and len(d40a):
        check(
            "aphakic pivot term far larger than phakic (pupil/N' coincidence breaks)",
            abs(d40a.iloc[0]["T_piv"]) > abs(d40p.iloc[0]["T_piv"]),
            f"phakic pivot={d40p.iloc[0]['T_piv']:+.3f}deg  aphakic pivot={d40a.iloc[0]['T_piv']:+.3f}deg",
        )

    print(
        f"\nnavarro experiment: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS NEED REVIEW (see WARN above)'}\n"
    )


EXPERIMENT_SIZES_DEV = {"synteyes": 50, "parameter_sweep": 50}
EXPERIMENT_SIZES_FULL = {"synteyes": 4000, "parameter_sweep": 4000}
SEEDS = {"synteyes": 42067, "parameter_sweep": 12345}


def _generate_specs(experiment: str, mode: RunMode) -> list[EyeSpec]:
    if experiment == "synteyes":
        n = (
            EXPERIMENT_SIZES_FULL["synteyes"]
            if mode == "full"
            else EXPERIMENT_SIZES_DEV["synteyes"]
        )
        return generate_synteyes_specs(
            n, SEEDS["synteyes"]
        ) + generate_pseudophakic_specs(mode)
    if experiment == "parameter_sweep":
        n = (
            EXPERIMENT_SIZES_FULL["parameter_sweep"]
            if mode == "full"
            else EXPERIMENT_SIZES_DEV["parameter_sweep"]
        )
        return generate_parameter_sweep_specs(n, SEEDS["parameter_sweep"])
    raise ValueError(f"Unknown experiment: {experiment!r}")


def _write_csv(df: pd.DataFrame, path: Path, *, allow_empty: bool = True) -> None:
    if not allow_empty and not len(df):
        return
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} rows to {path}")


def run(experiment: str, mode: RunMode, n_jobs: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    visisipy.set_backend("optiland", computation_backend="numpy")
    run_canary()
    run_retina_shape_canary()

    if experiment == "navarro":
        run_navarro(out_dir)
        return

    specs = _generate_specs(experiment, mode)
    n_eyes = len(specs)
    print(
        f"Running {experiment} ({mode} mode): {n_eyes} configurations, n_jobs={n_jobs}"
    )

    long_rows: list[dict] = []
    summary_rows: list[dict] = []
    chief_ray_rows: list[dict] = []
    failed_rows: list[dict] = []

    start_time = time.time()
    progress_every = max(1, n_eyes // 20)
    n_done = 0

    def _record(rows, summary, decomp_rows, failure) -> None:
        nonlocal n_done
        n_done += 1
        if failure is not None:
            failed_rows.append(failure)
        else:
            long_rows.extend(rows)
            summary_rows.append(summary)
            chief_ray_rows.extend(decomp_rows)
        if n_done % progress_every == 0 or n_done == n_eyes:
            elapsed = time.time() - start_time
            rate = n_done / elapsed if elapsed > 0 else 0.0
            eta = (n_eyes - n_done) / rate if rate > 0 else float("nan")
            print(
                f"  {n_done}/{n_eyes} processed ({elapsed:.1f}s elapsed, ~{eta:.1f}s remaining)"
            )

    if n_jobs <= 1:
        _init_worker()
        for spec in specs:
            _record(*process_eye(spec))
    else:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_jobs, initializer=_init_worker
        ) as executor:
            futures = [executor.submit(process_eye, spec) for spec in specs]
            for future in concurrent.futures.as_completed(futures):
                _record(*future.result())

    long_df = pd.DataFrame(long_rows)
    if len(long_df):
        sort_keys = [k for k in _LONG_ROW_SPEC_KEYS if k in long_df.columns] + ["theta"]
        long_df = long_df.sort_values(sort_keys).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df):
        sort_keys = [k for k in _LONG_ROW_SPEC_KEYS if k in summary_df.columns]
        if sort_keys:
            summary_df = summary_df.sort_values(sort_keys).reset_index(drop=True)
    failed_df = pd.DataFrame(failed_rows) if failed_rows else pd.DataFrame()

    decomp_df = pd.DataFrame(chief_ray_rows)
    if len(decomp_df):
        sort_keys = [k for k in _LONG_ROW_SPEC_KEYS if k in decomp_df.columns] + [
            "theta"
        ]
        decomp_df = decomp_df.sort_values(sort_keys).reset_index(drop=True)

    if experiment == "synteyes":
        # Phakic and pseudophakic rows go to separate files (different columns; merged on eye_id downstream).
        is_phakic = (
            summary_df["design"] == "phakic"
            if len(summary_df)
            else pd.Series(dtype=bool)
        )
        _write_csv(summary_df[is_phakic], out_dir / "synteyes_summary.csv")
        _write_csv(
            summary_df[~is_phakic], out_dir / "synteyes_pseudophakic_summary.csv"
        )
        is_phakic_long = (
            long_df["design"] == "phakic" if len(long_df) else pd.Series(dtype=bool)
        )
        _write_csv(long_df[is_phakic_long], out_dir / "synteyes_long.csv")
        _write_csv(long_df[~is_phakic_long], out_dir / "synteyes_pseudophakic_long.csv")

        phakic_failed_path = out_dir / "synteyes_failed.csv"
        pp_failed_path = out_dir / "synteyes_pseudophakic_failed.csv"
        if len(failed_df):
            is_phakic_failed = failed_df["design"] == "phakic"
            _write_csv(
                failed_df[is_phakic_failed], phakic_failed_path, allow_empty=False
            )
            _write_csv(failed_df[~is_phakic_failed], pp_failed_path, allow_empty=False)
        else:
            phakic_failed_path.unlink(missing_ok=True)
            pp_failed_path.unlink(missing_ok=True)

        if len(
            decomp_df
        ):  # phakic only: compute_chief_ray is set for phakic specs only
            _write_csv(decomp_df, out_dir / "synteyes_chief_ray_decomposition.csv")
    else:
        _write_csv(summary_df, out_dir / f"{experiment}_summary.csv")
        _write_csv(long_df, out_dir / f"{experiment}_long.csv")
        failed_path = out_dir / f"{experiment}_failed.csv"
        if len(failed_df):
            _write_csv(failed_df, failed_path, allow_empty=False)
        else:
            failed_path.unlink(missing_ok=True)
        if len(decomp_df):
            _write_csv(decomp_df, out_dir / f"{experiment}_chief_ray_decomposition.csv")

    if failed_rows:
        print(f"WARNING: {len(failed_rows)} configurations failed to build/solve.")

    total_time = time.time() - start_time
    print(
        f"\nTotal time: {total_time:.1f}s"
        + (f" ({total_time / n_eyes * 1000:.1f}ms/eye)" if n_eyes else "")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=["navarro", "synteyes", "parameter_sweep", "all"],
        required=True,
    )
    parser.add_argument("--mode", choices=["dev", "full"], default="dev")
    parser.add_argument("--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    experiments = (
        ["navarro", "synteyes", "parameter_sweep"]
        if args.experiment == "all"
        else [args.experiment]
    )
    for experiment in experiments:
        print(f"\n{'=' * 70}\n{experiment}\n{'=' * 70}")
        run(experiment, args.mode, args.n_jobs, args.out)


if __name__ == "__main__":
    main()
