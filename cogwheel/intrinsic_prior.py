"""
Priors for estimating only the intrinsic parameter of compact binary mergers
"""
from .prior import CombinedPrior
from .gw_prior import UniformDetectorFrameMassesPrior, FlatChieffPrior, ZeroInplaneSpinsPrior, ZeroTidalDeformabilityPrior

class IntrinsicParametersPrior(CombinedPrior):
    
    prior_classes = [UniformDetectorFrameMassesPrior,
                    FlatChieffPrior,
                    ZeroInplaneSpinsPrior,
                    ZeroTidalDeformabilityPrior]