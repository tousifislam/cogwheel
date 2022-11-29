import numpy as np 
from cogwheel.prior import Prior
from cogwheel.gw_prior.extrinisic import ReferenceDetectorMixin
from astropy import cosmology as cosmo
import astropy.units as units
from scipy.interpolate import splev, splrep

class InjectionMassPrior(ReferenceDetectorMixin, Prior):
    
    standard_params = ['m1', 'm2', 'd_luminosity']
    range_dic = {'m1_source': NotImplemented,
                 'lnq': NotImplemented,
                 'd_hat': NotImplemented}
                  
    reflective_params = ['lnq']
    conditioned_on = ['ra', 'dec', 'psi', 'iota']
    
    def __init__(self, *, tgps, ref_det_name, m1_source_range, alpha=1.0, q_min=.05, 
                 d_hat_max=500., symmetrize_lnq=False,**kwargs):
        lnq_min = np.log(q_min)
        self.alpha = alpha
        self.range_dic = {'m1_source': m1_source_range,
                          'lnq': (lnq_min, -lnq_min * symmetrize_lnq),
                          'd_hat' : (0, d_hat_max)}
        self.tgps = tgps
        self.ref_det_name=ref_det_name
        super().__init__(tgps=tgps, ref_det_name=ref_det_name, **kwargs)
        
    
    def _conversion_factor(self, ra, dec, psi, iota, m1, m2):
        """
        Return conversion factor such that
            d_luminosity = d_hat * conversion_factor.
        """
        mchirp = gw_utils.m1m2_to_mchirp(m1, m2)
        response = np.abs(self.geometric_factor_refdet(ra, dec, psi, iota))
        return mchirp**(5/6) * response

    
    def luminosity_distance_to_redshift(self, d_luminosity):
        """
        computes redshift from luminosity distances assuming Planck15 cosmology
        """
        cosmology = cosmo.Planck15
        return cosmo.z_at_value(cosmology.luminosity_distance, d_luminosity*units.Mpc)
    
    
    def redshift_to_luminosity_distance(self, redshift):
        """
        computes luminosity distances from redshiftassuming Planck15 cosmology
        """
        cosmology = cosmo.Planck15
        return cosmology.luminosity_distance(redshift).value

    
     def build_spline_for_distances(self, mchirp_source, lnq, d_hat, ra, dec, 
                                             psi, iota, redshift_min=None, redshift_max=None):
        """
        build spline fit for redshift as a function of d_hat conditioned on mchirp and
        other extrinsic params
        """
        # redshift ranges - take some fiducial values for now
        if redshift_min=None:
            redshift_min=0.01
        if redshift_max=None:
            redshift_max=1
        # initiate a grid for redshift
        redshift_grid = np.linspace(redshift_min, redshift_max, 1000) 
        # compute respective detector frame masses
        mchirp_grid = mchirp_source * (1+redshift_grid)
        eta = gw_utils.q_to_eta(np.exp(lnq))
        m1_grid, m2_grid = gw_utils.mchirpeta_to_m1m2(mchirp_grid, eta)
        # initiate a grid for d_hat
        d_hat_grid = np.linspace(range_dic['d_hat'][0], range_dic['d_hat'][1], 1000)
        # obtain d_luminosity values
        d_luminosity_grid = d_hat_grid * self._conversion_factor(ra, dec, psi,
                                                                iota, m1_grid, m2_grid)
        # build spline
        spl = splrep(d_hat_grid, d_luminosity_grid)
        
        return spl
    
    def mchirp_source_and_d_hat_to_redshift(self, mchirp_source, lnq, d_hat, ra, dec, psi, iota):
        """
        computes redshift and luminosity distances from mchirp_source and d_hat
        this is conditioned on other extrinsic params
        """
        # build spline fit for redshift as a function of d_hat
        spl = build_spline_for_distances(mchirp_source, lnq, d_hat, ra, dec, psi, iota)
        # compute redshift and luminosity distance from d_hat
        # using spline
        d_luminosity= splev(d_hat, spl)
        redshift  = self.luminosity_distance_to_redshift(d_luminosity)
        
        raise redshift, d_luminosity
        
        
    def transform(self, m1_source, lnq, d_hat, ra, dec, psi, iota):
        """
        to go from sampled params to standard params
        """
        m2_source = m1_source * np.exp(lnq)
        mchirp_source = gw_utils.m1m2_to_mchirp(m1_source, m2_source)
        redshift, d_luminosity = self.mchirp_source_and_d_hat_to_redshift(mchirp_source, lnq, d_hat, 
                                                                          ra, dec, psi, iota)
        m1 = m1_source * (1+redshift)
        
        return {'m1': m1,
                'm2': m1 * np.exp(lnq),
                'd_luminosity': d_luminosity}
    
    
    def inverse_transform(self, m1, m2, d_luminosity, ra, dec, psi, iota):
        """
        to go from standard params to sampled params
        """
        redshift = luminosity_distance_to_redshift(d_luminosity)
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
        m1 = par_dict['m1']
        m2 = par_dict['m2']
        
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
        
        # individual prior probability density
        lnprior_m1 = -self.alpha * np.log(m1_source)
        lnprior_q = lnq
        lnprior_d_hat = np.log(d_luminosity**3 / d_hat) 
        
        return lnprior_m1 + lnprior_q + lnprior_d_hat
        