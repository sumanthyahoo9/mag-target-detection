"""
All tunable knobs in one place. 
Change a value here and re-run.
"""
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    """
    This config has the tunable knobs.
    Any value can be changed here for testing. 
    """
    # --- Survey geometry ---
    patch_size_m: float = 50.0          # patch is 50 m x 50 m
    res_m: float = 0.5                  # grid pixel size -> 100 x 100 map
    line_spacing_m: float = 2.0         # distance between north-south survey lines
    sensor_alt_m: float = 2.0           # sensor height above the seabed

    # --- Earth's field (sets the shape of every anomaly) ---
    inclination_deg: float = 65.0       # field dips 65 deg below horizontal (mid-northern latitude)
    declination_deg: float = 0.0        # field points to geographic north

    # --- Targets: buried ferrous objects (one compact source each) ---
    n_targets: tuple = (0, 3)           # min, max per patch
    target_moment: tuple = (0.5, 5.0)   # magnetic moment, A·m^2
    target_burial_m: tuple = (0.0, 3.0) # depth below seabed
    target_remanence: float = 0.3       # 0 = purely induced (along Earth's field), 1 = random direction

    # --- Confounders: magnetic boulders (clusters of sources) ---
    n_boulders: tuple = (0, 4)
    boulder_sources: tuple = (2, 4)     # sources per boulder cluster
    boulder_moment: tuple = (0.2, 1.5)  # per source, A·m^2 (cluster total overlaps targets)
    boulder_radius_m: float = 1.2       # sources scattered within this radius
    boulder_burial_m: tuple = (0.0, 1.0)

    # --- Background and survey errors ---
    geology_nT: float = 15.0            # amplitude of broad geological variation
    drift_nT: float = 3.0               # step size of the line-to-line drift random walk
    noise_nT: float = 0.5               # sensor noise (std)

    # --- Labels ---
    label_sigma_px: float = 2.0         # Gaussian label width (2 px = 1 m)
    margin_m: float = 3.0               # keep objects away from patch edges

    # --- Dataset sizes and seeds (different seeds -> independent splits) ---
    n_train: int = 2000
    n_val: int = 300
    n_test: int = 500
    seeds: tuple = (0, 1, 2, 3)         # train, val, test, stress

    @property
    def grid_n(self) -> int:
        """
        Size of the grid
        """
        return int(round(self.patch_size_m / self.res_m))


# A harder setting for stress-testing generalisation
HARD = replace(Config(), target_remanence=0.8, target_burial_m=(1.0, 4.0),
               line_spacing_m=3.0, noise_nT=1.0, drift_nT=5.0)
