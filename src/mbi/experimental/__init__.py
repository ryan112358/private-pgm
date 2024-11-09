from .factor_graph import FactorGraph
from .region_graph import RegionGraph
from .local_inference import LocalInference
from .public_inference import PublicInference

try:
    from .mixture_inference import MixtureInference
except:
    import warnings

    warnings.warn("MixtureInference disabled, please install jax and jaxlib")

