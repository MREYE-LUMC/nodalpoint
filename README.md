# Optical origin of wide-field angular linearity in the human eye

Simulation code and data for

> J.-W. M. Beenakker, C. Haasjes and S. Habraken, *Optical origin of wide-field angular linearity in the human eye* (submitted, 2026).

When the retinal landing point of a chief ray is expressed as an angle α about the eye's second nodal point N′, α ≈ θ (the field angle) to within a degree out to far-peripheral eccentricities. This repository contains the ray-tracing experiments that test this mapping in the Navarro schematic eye, in 4,000 statistically generated (3D SyntEyes) phakic eyes, in the same eyes after implantation of two IOL designs, and in a 4,000-eye Latin-hypercube parameter sweep, together with the notebook that produces the manuscript's figures and reported values.

## Contents

| File | Purpose |
|---|---|
| `eye_sim.py` | Simulation pipeline |
| `synteyes.py` | Generates synthetic ocular biometry with the 3D SyntEyes model |
| `modeldata.json` | SyntEyes model parameters |
| `iol_designs.json` | The two IOL shape families used in the pseudophakic cohort. radii, thickness and conics as polynomials in a design parameter *x* that is solved per eye for on-axis emmetropia. |
| `share_data.ipynb` | Reproduces the manuscript figures and prints the values quoted in Table 1 and the Results. |
| `results.zip` | The simulation output used for the manuscript. Unzip to `results/`. |
| `requirements.txt` | Pinned package versions used for the manuscript. |

## Installation

Python ≥ 3.10 is required. In a fresh virtual environment:

```bash
pip install -r requirements.txt
```

The ray tracer is [Visisipy](https://github.com/MREYE-LUMC/visisipy) 0.4.0 with the NumPy backend of [Optiland](https://github.com/HarrisonKramer/optiland) 0.6.0; no commercial optical-design software is needed.

## Running the simulations

```bash
python eye_sim.py --experiment navarro                       # single-eye checks on the Navarro model (seconds)
python eye_sim.py --experiment synteyes        --mode full   # 4,000 phakic + 2 x 4,000 pseudophakic eyes
python eye_sim.py --experiment parameter_sweep --mode full   # 4,000-eye Latin-hypercube sweep
python eye_sim.py --experiment all             --mode full
```

Options:

- `--mode dev` (default) traces 50 eyes per cohort for a quick functional test; `--mode full` reproduces the manuscript cohorts.
- `--n-jobs N` sets the number of worker processes (default: all cores but one). - `--out DIR` changes the output directory (default `results/`).

Every run starts with two self-checks, and the `navarro` experiment prints PASS/FAIL for six analytic expectations.

Run the script from the repository root so that `synteyes.py`, `modeldata.json` and `iol_designs.json` are found.

## Output files

All angles are in degrees and lengths in millimetres.

| File | Content |
|---|---|
| `*_summary.csv` | One row per eye (and per IOL design for the pseudophakic cohort): input geometry, retina `R`/`Q`, paraxial second nodal point `z_N2`, corneal centre of curvature `z_corneaC`, exit pupil `z_EP`, unit-slope and minimum-distortion reference vertices, through-origin slope and R² of α vs θ (≤ 60°), `error_mm_40`, central and 40° spherical equivalent. |
| `*_long.csv` | One row per eye per field angle (0–80° in 5° steps). |
| `*_chief_ray_decomposition.csv` | Per eye and field angle, the exact split of `alpha − theta` into the angular demagnification term `T_ang`, the pivot term `T_piv` and the retinal-shape term `T_ret`. |
| `synteyes_pseudophakic_failed.csv` | Pseudophakic eyes for which no IOL power in the design family reached on-axis emmetropia. |
| `navarro_*.csv` | Single-surface exactness check, three-pivot comparison (pupil / N′ / retina centre), reference-vertex scans for the phakic and aphakic Navarro eye, corneal-asphericity sweep and chief-ray decomposition. |

## Reproducing the manuscript figures and numbers

Unzip `results.zip` (or run the simulations) so that `results/` exists, then open `share_data.ipynb` and run all cells. The notebook requires the full installation above because it imports constants from `eye_sim.py`.

## Related software and references

- Visisipy - paper: C. Haasjes, K. Harrison and J.-W. M. Beenakker, "Visisipy: open and accessible vision simulations in Python", Optica Open (2026).
- Visisipy - code: https://github.com/MREYE-LUMC/visisipy
- Optiland - code: https://github.com/HarrisonKramer/optiland
- SyntEyes: J. J. Rozema et al., Invest. Ophthalmol. Vis. Sci. 57, 683–691 (2016), doi:10.1167/iovs.16-20877.
- 3D SyntEyes retinal extension: N. van Dam et al., "Three-dimensional statistical eye models" (submitted, 2026).
- Nodal-point mapping in patient-specific eye models: C. Haasjes, T. H. K. Vu and J.-W. M. Beenakker, Med. Phys. (2025), doi:10.1002/mp.17576.
- Navarro wide-angle eye: I. Escudero-Sanz and R. Navarro, J. Opt. Soc. Am. A 16, 1881 (1999).

## License

Released under CC0 1.0 (see `LICENSE`).

## Contact

Jan-Willem Beenakker, Leiden University Medical Center — j.w.m.beenakker@lumc.nl 
