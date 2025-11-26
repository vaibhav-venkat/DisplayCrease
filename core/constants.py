"""Application-wide constants and enumerations."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ModelOption:
    label: str
    value: str
    description: str
    supports_ga: bool = False


MODEL_OPTIONS: List[ModelOption] = [
    ModelOption("Spheres", "spheres", "Monodisperse spherical scatterers"),
    ModelOption("Cylinders", "cylinders", "Rigid cylindrical scatterers"),
    ModelOption("Chained cylinders", "chained_cylinders", "Cylinders connected in series"),
    ModelOption("Ellipsoids", "ellipsoids", "Axisymmetric ellipsoidal scatterers", supports_ga=True),
    ModelOption("Hollow tubes", "hollow_tubes", "Anisotropic hollow tubes", supports_ga=True),
    ModelOption("Super Ellipsoids", "super_ellipsoids", "Generalized ellipsoid family"),
    ModelOption("Vesicles", "vesicles", "Spherical vesicle structures"),
    ModelOption("Patchy Vesicles", "patchy_vesicles", "Vesicles with surface heterogeneity"),
]

MODEL_VALUE_TO_LABEL = {option.value: option.label for option in MODEL_OPTIONS}
MODEL_LABEL_TO_VALUE = {option.label: option.value for option in MODEL_OPTIONS}

SUPPORTED_OPTIMIZATION_METHODS: List[Dict[str, str]] = [
    {"label": "Genetic Algorithm", "value": "genetic_algorithm"},
    {"label": "Bayesian Optimization", "value": "bayesian_optimization"},
    {"label": "Particle Swarm Optimization", "value": "particle_swarm"},
    {"label": "Reinforcement Learning", "value": "reinforcement_learning"},
]

DEFAULT_DATA_CONFIG = {
    "phi_min": 0.0,
    "phi_max": 180.0,
    "q_min_exp": -2.1,
    "q_max_exp": -0.9,
    "is_log_transformed": False,
}

GA_SUPPORTED_MODELS = {option.value for option in MODEL_OPTIONS if option.supports_ga}
GA_OUTPUT_ROOT = "outputs/ga_runs"
GA_PLOT_ROOT = "outputs/plots"

MODEL_PARAMETER_INFO: Dict[str, List[str]] = {
    "spheres": [
        "Radius distribution",
        "Volume fraction",
        "Scattering length density contrast",
    ],
    "cylinders": [
        "Cylinder radius",
        "Aspect ratio",
        "Orientation distribution",
    ],
    "chained_cylinders": [
        "Segment length",
        "Segment radius",
        "Chain flexibility",
    ],
    "ellipsoids": [
        "Mean radius",
        "Radius dispersion",
        "Aspect ratio",
        "Aspect ratio dispersion",
        "Orientation parameter κ",
        "Volume fraction",
    ],
    "hollow_tubes": [
        "Mean tube diameter",
        "Eccentricity mean",
        "Eccentricity std deviation",
        "Orientation angle",
        "κ exponent",
        "Cone angle",
        "Herd tube diameter",
        "Herd tube length",
        "Extra herd nodes",
    ],
    "super_ellipsoids": [
        "Principal axes",
        "Shape exponent",
        "Orientation distribution",
    ],
    "vesicles": [
        "Shell radius",
        "Shell thickness",
        "Inner/outer SLD",
    ],
    "patchy_vesicles": [
        "Patch coverage",
        "Patch SLD",
        "Background SLD",
    ],
}

# Accepted upload suffixes for matrix-like files
ALLOWED_EXTENSIONS = {"txt", "csv", "dat"}
