from dataclasses import dataclass


@dataclass
class YProfileConfig:
    alpha: float
    beta0: float  # direct normalization for beta
    gamma: float
    log10_P0: float  # log10 of absolute P0 normalization
