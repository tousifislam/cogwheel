"""Generate strain waveforms and project them onto detectors."""

from collections import defaultdict
import numpy as np

import lal
import lalsimulation

from . import gw_utils
from . import utils

ZERO_INPLANE_SPINS = {'s1x': 0.,
                      's1y': 0.,
                      's2x': 0.,
                      's2y': 0.}

DEFAULT_PARS = {**ZERO_INPLANE_SPINS,
                's1z': 0.,
                's2z': 0.,
                'l1': 0.,
                'l2': 0.}

APPROXIMANTS = {}


class Approximant:
    """Bookkeeping of LAL approximants' metadata."""
    def __init__(self, approximant: str, harmonic_modes: list,
                 aligned_spins: bool, tides: bool):
        self.approximant = approximant
        self.harmonic_modes = harmonic_modes
        self.aligned_spins = aligned_spins
        self.tides = tides
        APPROXIMANTS[approximant] = self


Approximant('IMRPhenomD', [(2, 2)], True, False)
#Approximant('IMRPhenomXPHM', [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4)], False,
#            False)
Approximant('IMRPhenomXPHM', [(2, 2)], False,
            False)


def out_of_bounds(par_dic):
    """
    Return whether parameters in `par_dic` are out of physical bounds.
    """
    return (any(par_dic[positive] < 0 for positive in
                ['m1', 'm2', 'd_luminosity', 'l1', 'l2', 'iota'])
            or any(np.linalg.norm(s) > 1 for s in [
                [par_dic['s1x'], par_dic['s1y'], par_dic['s1z']],
                [par_dic['s2x'], par_dic['s2y'], par_dic['s2z']]])
            or par_dic['iota'] > np.pi
            or np.abs(par_dic['dec'] > np.pi/2))


def compute_hplus_hcross(f_ref, f, par_dic, approximant: str,
                         harmonic_modes=None):
    """
    Generate frequency domain waveform using LAL.
    Return hplus, hcross evaluated at f.

    Parameters
    ----------
    approximant: String with the approximant name.
    f_ref: Reference frequency in Hz
    f: Frequency array in Hz
    par_dic: Dictionary of source parameters. Needs to have these keys:
                 m1, m2, d_luminosity, iota, vphi;
             plus, optionally:
                 s1x, s1y, s1z, s2x, s2y, s2z, l1, l2.
    harmonic_modes: Optional, list of 2-tuples with (l, m) pairs
                  specifying which (co-precessing frame) higher-order
                  modes to include.
    """
    # Parameters ordered for lalsimulation.SimInspiralChooseFDWaveformSequence
    lal_params = [
        'vphi', 'm1_kg', 'm2_kg', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z',
        'f_ref', 'd_luminosity_meters', 'iota', 'lal_dic', 'approximant', 'f']

    par_dic = DEFAULT_PARS | par_dic

    # SI unit conversions
    par_dic['d_luminosity_meters'] = par_dic['d_luminosity'] * 1e6 * lal.PC_SI
    par_dic['m1_kg'] = par_dic['m1'] * lal.MSUN_SI
    par_dic['m2_kg'] = par_dic['m2'] * lal.MSUN_SI

    lal_dic = lal.CreateDict()  # Contains tidal and higher-mode parameters
    # Tidal parameters
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda1(
        lal_dic, par_dic['l1'])
    lalsimulation.SimInspiralWaveformParamsInsertTidalLambda2(
        lal_dic, par_dic['l2'])
    # Higher-mode parameters
    if harmonic_modes is not None:
        mode_array = lalsimulation.SimInspiralCreateModeArray()
        for l, m in harmonic_modes:
            lalsimulation.SimInspiralModeArrayActivateMode(mode_array, l, m)
        lalsimulation.SimInspiralWaveformParamsInsertModeArray(lal_dic,
                                                               mode_array)
    par_dic['lal_dic'] = lal_dic

    par_dic['approximant'] = lalsimulation.GetApproximantFromString(
        approximant)

    par_dic['f_ref'] = f_ref

    f0_is_0 = f[0] == 0  # In this case we will set h(f=0) = 0
    par_dic['f'] = lal.CreateREAL8Sequence(len(f))
    par_dic['f'].data = f
    if f0_is_0:
        par_dic['f'].data[0] = par_dic['f'].data[1]

    try:
        hplus, hcross = lalsimulation.SimInspiralChooseFDWaveformSequence(
            *[par_dic[par] for par in lal_params])
    except:
        print('Error while calling LAL at these parameters:', par_dic)
        raise
    hplus_hcross = np.stack([hplus.data.data, hcross.data.data])
    if f0_is_0:
        hplus_hcross[:, 0] = 0
    return hplus_hcross


class WaveformGenerator(utils.JSONMixin):
    """
    Class that provides methods for generating frequency domain
    waveforms, in terms of `hplus, hcross` or projected onto detectors.
    "Fast" and "slow" parameters are distinguished: the last waveform
    calls are cached and can be computed fast when only fast parameters
    are changed.
    The boolean attribute `disable_precession` can be set to ignore
    inplane spins.
    """
    params = sorted(['d_luminosity', 'dec', 'iota', 'l1', 'l2', 'm1', 'm2',
                     'psi', 'ra', 's1x', 's1y', 's1z', 's2x', 's2y', 's2z',
                     't_geocenter', 'vphi'])

    fast_params = sorted(['d_luminosity', 'dec', 'psi', 'ra', 't_geocenter', 'vphi'])
    slow_params = sorted(set(params) - set(fast_params))

    _projection_params = sorted(['dec', 'psi', 'ra', 't_geocenter'])
    _waveform_params = sorted(set(params) - set(_projection_params))

    _s1xy_inds = list(map(slow_params.index, ['s1x', 's1y']))
    _s2xy_inds = list(map(slow_params.index, ['s2x', 's2y']))

    def __init__(self, detector_names, tgps, tcoarse, approximant, f_ref,
                 harmonic_modes=None, disable_precession=False,
                 n_cached_waveforms=1):
        super().__init__()

        self.detector_names = detector_names
        self.tgps = tgps
        self.tcoarse = tcoarse
        self._approximant = approximant
        self.harmonic_modes = harmonic_modes
        self.disable_precession = disable_precession
        self.f_ref = f_ref
        self.n_cached_waveforms = n_cached_waveforms

        self.n_slow_evaluations = 0
        self.n_fast_evaluations = 0

    @property
    def approximant(self):
        """String with waveform approximant name."""
        return self._approximant

    @approximant.setter
    def approximant(self, app: str):
        """
        Set `approximant` and reset `harmonic_modes` per
        `APPROXIMANTS[approximant].harmonic_modes`; print a warning that
        this was done.
        Raise `ValueError` if `APPROXIMANTS` does not contain the
        requested approximant.
        """
        if app not in APPROXIMANTS:
            raise ValueError(f'Add {app} to `waveform.APPROXIMANTS`.')
        self._approximant = app
        self.harmonic_modes = None
        print(f'`approximant` changed to {app}, setting `harmonic_modes` to '
              f'{self.harmonic_modes}.')

    @property
    def harmonic_modes(self):
        """List of `(l, m)` pairs."""
        return self._harmonic_modes

    @harmonic_modes.setter
    def harmonic_modes(self, harmonic_modes):
        """
        Set `self._harmonic_modes` implementing defaults based on the
        approximant, this requires hardcoding which modes are
        implemented by each approximant; it is a necessary evil because
        we need to know `m` in order to make `vphi` a fast parameter.
        Also set `self._harmonic_modes_by_m` with a dictionary whose
        keys are `m` and whose values are a list of `(l, m)` tuples with
        that `m`.
        """
        self._harmonic_modes \
            = harmonic_modes or APPROXIMANTS[self.approximant].harmonic_modes

        self._harmonic_modes_by_m = defaultdict(list)
        for l, m in self._harmonic_modes:
            self._harmonic_modes_by_m[m].append((l, m))

    @property
    def n_cached_waveforms(self):
        """Nonnegative integer, number of cached waveforms."""
        return self._n_cached_waveforms

    @n_cached_waveforms.setter
    def n_cached_waveforms(self, n_cached_waveforms):
        self.cache = [{'slow_par_vals': np.array(np.nan),
                       'approximant': None,
                       'f_ref': None,
                       'f': None,
                       'harmonic_modes_by_m': {},
                       'hplus_hcross_0': None}
                      for _ in range(n_cached_waveforms)]
        self._n_cached_waveforms = n_cached_waveforms

    def get_strain_at_detectors(self, f, par_dic, by_m=False):
        """
        Get strain measurable at detectors.

        Parameters
        ----------
        f: 1d array of frequencies [Hz]
        par_dic: parameter dictionary per `WaveformGenerator.params`.
        by_m: bool, whether to return waveform separated by `m`
              harmonic mode (summed over `l`), or already summed.

        Return
        ------
        n_detectors x n_frequencies array with strain at detector.
        If by_m, output is (n_m x n_detectors x n_frequencies)
        """
        waveform_par_dic = {par: par_dic[par] for par in self._waveform_params}

        # hplus_hcross shape: (n_m x 2 x n_frequencies), n_m optional
        hplus_hcross = self.get_hplus_hcross(f, waveform_par_dic, by_m)

        # fplus_fcross shape: (2 x n_detectors)
        fplus_fcross = np.array(gw_utils.fplus_fcross(
            self.detector_names, par_dic['ra'], par_dic['dec'], par_dic['psi'],
            self.tgps))

        time_delays = gw_utils.time_delay_from_geocenter(
            self.detector_names, par_dic['ra'], par_dic['dec'], self.tgps)

        # shifts shape: (n_detectors x n_frequencies)
        shifts = np.exp(-2j*np.pi * f * (self.tcoarse
                                         + par_dic['t_geocenter']
                                         + time_delays[:, np.newaxis]))

        # Detector strain (n_m x n_detectors x n_frequencies), n_m optional
        return np.sum(fplus_fcross[..., np.newaxis]
                      * hplus_hcross[..., np.newaxis, :], axis=-3) * shifts

    def get_hplus_hcross(self, f, waveform_par_dic, by_m=False):
        """
        Return hplus, hcross waveform strain.
        Note: inplane spins will be zeroized if `self.disable_precession`
              is `True`.

        Parameters
        ----------
        f: 1d array of frequencies [Hz]
        waveform_par_dic: dictionary per
                          `WaveformGenerator._waveform_params`.
        by_m: bool, whether to return harmonic modes separately by m (l
              summed over) or all modes already summed over.

        Return
        ------
        array with (hplus, hcross), of shape `(2, len(f))` if `by_m` is
        `False`, or `(n_m, 2, len(f))` if `by_m` is `True`, where `n_m`
        is the number of harmonic modes with different `m`.
        """
        if self.disable_precession:
            waveform_par_dic.update(ZERO_INPLANE_SPINS)

        slow_par_vals = np.array([waveform_par_dic[par]
                                  for par in self.slow_params])
        self._rotate_inplane_spins(slow_par_vals, waveform_par_dic['vphi'])

        # Attempt to use cached waveform for fast evaluation:
        matching_cache = self._matching_cache(slow_par_vals, f)
        if matching_cache:
            hplus_hcross_0 = matching_cache['hplus_hcross_0']
            self.n_fast_evaluations += 1
        else:
            # Compute the waveform mode by mode and update cache.
            waveform_par_dic_0 = dict(zip(self.slow_params, slow_par_vals),
                                      d_luminosity=1., vphi=0.)
            # hplus_hcross_0 is a (n_m x 2 x n_frequencies) array with
            # sum_l (hlm+, hlmx), at vphi=0, d_luminosity=1Mpc.
            hplus_hcross_0 = np.array(
                [compute_hplus_hcross(self.f_ref, f, waveform_par_dic_0,
                                      self.approximant, modes)
                 for modes in self._harmonic_modes_by_m.values()])
            cache_dic = {'approximant': self.approximant,
                         'f_ref': self.f_ref,
                         'f': f,
                         'slow_par_vals': slow_par_vals,
                         'harmonic_modes_by_m': self._harmonic_modes_by_m,
                         'hplus_hcross_0': hplus_hcross_0}
            # Append new cached waveform and delete oldest
            self.cache.append(cache_dic)
            self.cache.pop(0)

            self.n_slow_evaluations += 1

        # hplus_hcross is a (n_m x 2 x n_frequencies) array.
        m_arr = np.array(list(self._harmonic_modes_by_m)).reshape((-1, 1, 1))
        hplus_hcross = (np.exp(1j * m_arr * waveform_par_dic['vphi'])
                        / waveform_par_dic['d_luminosity'] * hplus_hcross_0)
        if by_m:
            return hplus_hcross
        return np.sum(hplus_hcross, axis=0)

    def _rotate_inplane_spins(self, slow_par_vals, vphi):
        """
        Rotate inplane spins (s1x, s1y) and (s2x, s2y) by an angle `vphi`,
        inplace in `slow_par_vals`.
        `slow_par_vals` must be a numpy array whose values correspond to
        `self.slow_params`.
        """
        sin_vphi = np.sin(vphi)
        cos_vphi = np.cos(vphi)
        rotation = np.array([[cos_vphi, -sin_vphi],
                             [sin_vphi, cos_vphi]])
        slow_par_vals[self._s1xy_inds] = (rotation
                                          @ slow_par_vals[self._s1xy_inds])
        slow_par_vals[self._s2xy_inds] = (rotation
                                          @ slow_par_vals[self._s2xy_inds])
        # dic['s1x'], dic['s1y'] = rotation @ np.array([dic['s1x'], dic['s1y']])
        # dic['s2x'], dic['s2y'] = rotation @ np.array([dic['s2x'], dic['s2y']])

    def _matching_cache(self, slow_par_vals, f, eps=1e-6):
        """
        Return entry of the cache that matches the requested waveform, or
        `False` if none of the cached waveforms matches that requested.
        """
        for cache_dic in self.cache[ : : -1]:
            if (np.linalg.norm(slow_par_vals-cache_dic['slow_par_vals']) < eps
                    and cache_dic['approximant'] == self.approximant
                    and cache_dic['f_ref'] == self.f_ref
                    and np.array_equal(cache_dic['f'], f)
                    and (cache_dic['harmonic_modes_by_m']
                         == self._harmonic_modes_by_m)):
                return cache_dic

        return False
