"""
Custom astrophysical population priors
"""
import numpy as np
from cogwheel import gw_utils, cosmology
from cogwheel.cosmology import comoving_to_luminosity_diff_vt_ratio

from cogwheel.prior import Prior, CombinedPrior
from cogwheel.gw_prior.extrinsic import (UniformPhasePrior,
                                        IsotropicInclinationPrior,
                                        IsotropicSkyLocationPrior,
                                        UniformTimePrior,
                                        UniformPolarizationPrior,
                                        ReferenceDetectorMixin)
from cogwheel.gw_prior.combined import RegisteredPriorMixin
from cogwheel.gw_prior.miscellaneous import (ZeroTidalDeformabilityPrior,
                                             FixedReferenceFrequencyPrior)
from cogwheel.gw_prior.spin import UniformEffectiveSpinPrior, ZeroInplaneSpinsPrior
from scipy import interpolate

class InjectionMassPrior(ReferenceDetectorMixin, Prior):
    """
    Prior class for 
        f(m1_source, q, chi_eff, D_L) = m1_source^alpha * D_L^2
    """
    standard_params = ['m1', 'm2', 'd_luminosity']
    range_dic = {'m1_source': NotImplemented,
                 #'lnq': NotImplemented,
                 'cum_q': (0,1),
                 'd_hat': NotImplemented}
                  
    conditioned_on = ['ra', 'dec', 'psi', 'iota']
    
    def __init__(self, *, tgps, ref_det_name, m1_source_range, alpha=2.0, m_min=1, 
                 d_hat_max=500., **kwargs):
        
        self.m_min = m_min
        self.alpha = alpha
        self.range_dic = {'m1_source': m1_source_range,
                          'cum_q': (0,1),
                          #'lnq': (lnq_min, -lnq_min * symmetrize_lnq),
                          'd_hat' : (0, d_hat_max)}
        
        # build an inverse spline for d_luminosity and d_hat
        # for future use
        # some reasonable values for the distance range 
        # - should be checked later
        d_luminosity_grid = np.linspace(0., 1000000., 10000)
        f_grid = self._f_of_d_luminosity(d_luminosity_grid) 
        self.fmax = max(f_grid)
        self._d_luminosity_of_f = interpolate.interp1d(f_grid, d_luminosity_grid)
        
        self.tgps = tgps
        self.ref_det_name = ref_det_name
        super().__init__(tgps=tgps, ref_det_name=ref_det_name, **kwargs)

    @staticmethod
    def _f_of_d_luminosity(d_luminosity):
        """
        function for f(d_L) = d_L / (1+z)^(5/6)
        """
        redshift = cosmology.z_of_d_luminosity(d_luminosity)
        return d_luminosity / (1+redshift)**(5/6)
        
        
    def _response_factor(self, ra, dec, psi, iota):
        """
        Return detector response function
        Eq (9) of https://arxiv.org/pdf/2207.03508.pdf
        """
        response = np.abs(self.geometric_factor_refdet(ra, dec, psi, iota))
        return response
    
    
    def _conversion_factor(self, ra, dec, psi, iota, m1, m2):
        """
        Return conversion factor such that
            d_luminosity = d_hat * conversion_factor.
        """
        mchirp = gw_utils.m1m2_to_mchirp(m1, m2)
        response = self._response_factor(ra, dec, psi, iota)
        return mchirp**(5/6) * response

        
    def transform(self, m1_source, cum_q, d_hat, ra, dec, psi, iota):
        """
        to go from sampled params to standard params
        """
        
        q_min = self.m_min / m1_source
        q_rng = 1 - q_min
        q = q_min + q_rng * cum_q
    
        m2_source = m1_source * q
        mchirp_source = gw_utils.m1m2_to_mchirp(m1_source, m2_source)
        R_k = self._response_factor(ra, dec, psi, iota)
        func_val = d_hat * mchirp_source**(5/6) * R_k
        # use inverse spline here
        if func_val>self.fmax:
            print(mchirp_source, R_k, m1_source, lnq, d_hat, ra, dec, psi, iota)
        d_luminosity = self._d_luminosity_of_f(func_val)[()]
        redshift = cosmology.z_of_d_luminosity(d_luminosity)
        m1 = m1_source * (1+redshift)
        
        return {'m1': m1,
                'm2': m1 * q,
                'd_luminosity': d_luminosity}
    
    
    def inverse_transform(self, m1, m2, d_luminosity, ra, dec, psi, iota):
        """
        to go from standard params to sampled params
        """
        redshift = cosmology.z_of_d_luminosity(d_luminosity)
        m1_source = m1 / (1+redshift)
        q_min = self.m_min / m1_source
        q_rng = 1 - q_min
        cum_q = (m2/m1 - q_min) / q_rng
    
        return {'m1_source': m1 / (1+redshift) ,
                'cum_q': cum_q,
                'd_hat': d_luminosity / self._conversion_factor(ra, dec, psi,
                                                                iota, m1, m2)}
    
    def lnprior(self, m1_source, cum_q, d_hat, ra, dec, psi, iota):
        """
        Prior distribution as defined in Eq(24) of https://arxiv.org/pdf/2008.07014.pdf
        f (m1_source, q, chi_eff, D_L) = m1_source^alpha * D_L^2
        Truncated power-law in the primary source-frame mass m1_source.
        Uniform prior in mass ratio q and spin chi_eff.
        Power law in the luminosity distance.
        """
        par_dict = self.transform(m1_source, cum_q, d_hat, ra, dec, psi, iota)
        d_luminosity = par_dict['d_luminosity']
        
        # individual prior probability density : power law in source frame mass
        lnprior_m1_source = -self.alpha * np.log(m1_source)
        # individual prior probability density : power law in luminosity distance
        lnprior_d_hat = np.log(d_luminosity**3 / d_hat
                               * comoving_to_luminosity_diff_vt_ratio(d_luminosity))
        # joint probability density
        lnprior_tot = lnprior_m1_source + lnprior_d_hat
        
        return lnprior_tot 
    
    def get_init_dict(self):
        """
        Return dictionary with keyword arguments to reproduce the class
        instance.
        """
        return  {'tgps': self.tgps, 
                 'ref_det_name': self.ref_det_name, 
                 'm1_source_range': self.range_dic['m1_source'], 
                 'alpha': self.alpha,
                 'm_min': self.m_min, 
                 'd_hat_max': self.range_dic['d_hat'][1]}
            
        
        
class PopulationPrior(RegisteredPriorMixin, CombinedPrior):
    """
    Prior class for sampling from an astrophysical mass/spin distribution
    following Roulet et al, https://arxiv.org/pdf/2008.07014.pdf
    """
    prior_classes = [IsotropicInclinationPrior,
                     IsotropicSkyLocationPrior,
                     UniformTimePrior,
                     UniformPolarizationPrior,
                     UniformPhasePrior,
                     InjectionMassPrior,
                     UniformEffectiveSpinPrior,
                     ZeroInplaneSpinsPrior,
                     ZeroTidalDeformabilityPrior,
                     FixedReferenceFrequencyPrior]

       