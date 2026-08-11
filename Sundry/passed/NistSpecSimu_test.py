import numpy as np
import math

# 常数
h = 4.135667696e-15  # eV·s (Planck constant in eV*s)
c = 2.99792458e8     # m/s
kB = 8.617333262145e-5  # eV/K

def ensure_units(wl, wl_unit='nm', E=None, E_unit='eV'):
    """Return wavelength in meters and E in eV (if provided)."""
    if wl_unit.lower() in ['nm','nanometer','nanometers']:
        wl_m = np.array(wl) * 1e-9
    elif wl_unit.lower() in ['um','µm','micron','microm']:
        wl_m = np.array(wl) * 1e-6
    elif wl_unit.lower() in ['ang','angstrom','å','a']:
        wl_m = np.array(wl) * 1e-10
    elif wl_unit.lower() in ['m','meter','meters']:
        wl_m = np.array(wl)
    else:
        raise ValueError('unknown wl_unit')

    E_eV = None
    if E is not None:
        if E_unit.lower() in ['ev','eV']:
            E_eV = np.array(E, dtype=float)
        elif E_unit.lower() in ['cm-1','cm^-1','1/cm']:
            E_eV = np.array(E, dtype=float) * 1.239841984e-4
        else:
            raise ValueError('unknown E_unit')
    return wl_m, E_eV

def compute_relative_intensities(wl, A, E, g, T,
                                 wl_unit='nm', E_unit='eV',
                                 ion_fraction=1.0,
                                 output_mode='photon',   # 'photon' or 'energy'
                                 normalization='sum'     # 'sum' or 'max' or None
                                ):
    """
    output_mode:
      - 'photon' => I_i ∝ N_upper * A  (photon flux)
      - 'energy' => I_i ∝ N_upper * A * h*nu  (energy flux)
    ion_fraction: fraction of atoms in this ionization stage (0..1)
    normalization: 'sum' => divide by sum, 'max' => divide by max, None => raw
    """
    wl_m, E_eV = ensure_units(wl, wl_unit=wl_unit, E=E, E_unit=E_unit)
    # level population (Boltzmann within this ion state)
    # U(T): partition function approximated by sum over provided levels
    U_vals = np.array(g) * np.exp(-np.array(E_eV) / (kB * T))
    U_sum = np.sum(U_vals)
    # upper level populations (proportional)
    N_upper_rel = (np.array(g) * np.exp(-np.array(E_eV) / (kB * T))) / U_sum
    # multiply by ion_fraction
    N_upper_rel = ion_fraction * N_upper_rel

    # Einstein A (s^-1) should be provided
    A = np.array(A, dtype=float)
    # photon flux proportional to N_upper * A
    I_photon = N_upper_rel * A

    if output_mode == 'photon':
        I = I_photon
    elif output_mode == 'energy':
        # hv in eV -> multiply by eV to get energy flux proportional
        nu = c / wl_m   # Hz
        h_eVs = h       # eV*s
        hv = h_eVs * nu # eV
        I = I_photon * hv
    else:
        raise ValueError('output_mode must be photon or energy')

    # normalization
    if normalization == 'sum':
        denom = np.sum(I)
        if denom != 0:
            I = I / denom
    elif normalization == 'max':
        m = np.max(I)
        if m != 0:
            I = I / m
    # else leave raw

    return I
