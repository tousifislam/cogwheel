"""
Priors for estimating only the intrinsic parameter of compact binary
mergers.
"""
from cogwheel.prior import CombinedPrior
from cogwheel.gw_prior.combined import RegisteredPriorMixin
from cogwheel.gw_prior.mass import UniformDetectorFrameMassesPrior
from cogwheel.gw_prior.tides import UniformTidalDeformabilitiesBNSPrior
from cogwheel.gw_prior.spin import (UniformEffectiveSpinPrior,
                                    IsotropicSpinsAlignedComponentsPrior,
                                    IsotropicHighSpinsAlignedComponentsPrior,
                                    ZeroInplaneSpinsPrior)
from cogwheel.gw_prior.miscellaneous import (ZeroTidalDeformabilityPrior,
                                             FixedReferenceFrequencyPrior)

class IntrinsicParametersPrior(RegisteredPriorMixin, CombinedPrior):
    """For BBH systems."""
    prior_classes = [UniformDetectorFrameMassesPrior,
                     UniformEffectiveSpinPrior,
                     ZeroInplaneSpinsPrior,
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior]

class IntrinsicTidalPrior(RegisteredPriorMixin, CombinedPrior):
    """For BNS systems."""
    prior_classes = [UniformTidalDeformabilitiesBNSPrior,
                     UniformDetectorFrameMassesPrior,
                     UniformEffectiveSpinPrior,
                     ZeroInplaneSpinsPrior,
                     FixedReferenceFrequencyPrior]
    
class LVKIntrinsicTidalPrior(RegisteredPriorMixin, CombinedPrior):
    """For BNS systems."""
    prior_classes = [UniformTidalDeformabilitiesBNSPrior,
                     UniformDetectorFrameMassesPrior,
                     IsotropicSpinsAlignedComponentsPrior,
                     ZeroInplaneSpinsPrior,
                     FixedReferenceFrequencyPrior]
    
class LVKHighSpinTidalPrior(RegisteredPriorMixin, CombinedPrior):
    """For BNS systems."""
    prior_classes = [UniformTidalDeformabilitiesBNSPrior,
                     UniformDetectorFrameMassesPrior,
                     IsotropicHighSpinsAlignedComponentsPrior,
                     ZeroInplaneSpinsPrior,
                     FixedReferenceFrequencyPrior]
