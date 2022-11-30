import numpy as np 
from cogwheel import gw_utils, cosmology
from cogwheel.prior import Prior
from cogwheel.gw_prior.extrinisic import ReferenceDetectorMixin
from scipy import interpolate

class InjectionMassPrior(ReferenceDetectorMixin, Prior):
    
    standard_params = ['m1', 'm2', 'd_luminosity']
    range_dic = {'m1_source': NotImplemented,
                 'lnq': NotImplemented,
                 'd_hat': NotImplemented}
                  
    reflective_params = ['lnq']
    conditioned_on = ['ra', 'dec', 'psi', 'iota']
    
    def __init__(self, *, tgps, ref_det_name, m1_source, lnq, ra, dec, psi, iota, 
                 m1_source_range, alpha=1.0, q_min=.05, 
                 d_hat_max=500., symmetrize_lnq=False,**kwargs):
        
        lnq_min = np.log(q_min)
        self.alpha = alpha
        self.range_dic = {'m1_source': m1_source_range,
                          'lnq': (lnq_min, -lnq_min * symmetrize_lnq),
                          'd_hat' : (0, d_hat_max)}
        
        # build an inverse spline for d_luminosity and d_hat
        # for future use
        m2_source = m1_source * np.exp(lnq)
        d_luminosity_grid = np.linspace(1., 100000., 10000)
        mchirp_source = gw_utils.m1m2_to_mchirp(m1_source, m2_source)
        R_k = _response_factor(ra, dec, psi, iota)
        d_hat_grid = self._f_of_d_luminosity(d_luminosity_grid) / (mchirp_source**(5/6) * R_k)
        self._d_luminosity_of_d_hat = interpolate.interp1d(d_hat_grid, d_luminosity_grid)
        
        self.tgps = tgps
        self.ref_det_name=ref_det_name
        super().__init__(tgps=tgps, ref_det_name=ref_det_name, **kwargs)

    
    def _f_of_d_luminosity(self, d_luminosity):
        """
        function for f(d_L) = d_L / (1+z)^(5/6)
        """
        redshift = cosmology.z_of_luminosity( d_luminosity)
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

        
    def transform(self, m1_source, lnq, d_hat, ra, dec, psi, iota):
        """
        to go from sampled params to standard params
        """
        # use inverse spline here
        d_luminosity = self._d_luminosity_of_d_hat(d_hat)
        redshift = cosmology.z_of_d_luminosity(d_luminosity)
        m1 = m1_source * (1+redshift)
        
        return {'m1': m1,
                'm2': m1 * np.exp(lnq),
                'd_luminosity': d_luminosity}
    
    
    def inverse_transform(self, m1, m2, d_luminosity, ra, dec, psi, iota):
        """
        to go from standard params to sampled params
        """
        redshift = cosmology.z_of_d_luminosity(d_luminosity)
        return {'m1_source': m1 / (1+redshift) ,
                'lnq': np.log(m2/m1),
                'd_hat': d_luminosity / self._conversion_factor(ra, dec, psi,
                                                                iota, m1, m2)}
    
    def lnprior(self, m1_source, lnq, d_hat, ra, dec, psi, iota):
        """
        Prior distrbution as defined in Eq(24) in https://arxiv.org/pdf/2008.07014.pdf
        Truncated power-law in the primary source-frame mass m1_source.
        Uniform prior in mass ratio q and spin chi_eff.
        Power law in the luminosity distance.
        """
        par_dict = self.transform(m1_source, lnq, d_hat, ra, dec, psi, iota)
        d_luminosity = par_dict['d_luminosity']
         
        # checks to make sure input params are not outside allowed ranges
        if lnq<self.range_dic['lnq'][0]:
            return -np.inf
        elif lnq>self.range_dic['lnq'][1]:
            return -np.inf
        elif m1_source<self.range_dic['m1_source'][0]:
            return -np.inf
        elif m1_source>self.range_dic['m1_source'][1]:
            return -np.inf
        elif d_hat<self.range_dic['d_hat'][0]:
            return -np.inf
        elif d_hat>self.range_dic['d_hat'][1]:
            return -np.inf
        
        # individual prior probability density : power law in source frame mass
        lnprior_m1_source = -self.alpha * np.log(m1_source)
        # individual prior probability density : uniform in mass ratio
        lnprior_lnq = lnq
        # individual prior probability density : power law in luminosity distance
        lnprior_d_hat = np.log(d_luminosity**3 / d_hat) 
        # joint probability density
        lnprior_tot = lnprior_m1_source + lnprior_lnq + lnprior_d_hat
        
        return lnprior_tot 
        