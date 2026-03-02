import numpy as np
import pandas as pd
from feos import Parameters, PureRecord, Identifier, EquationOfState, State
# from si_units import *
# Correcting imports to use feos.si if available, but since we installed si-units package,
# it might be available as 'si_units' or 'feos.si'
from si_units import KELVIN, BAR, RGAS, JOULE, MOL, KILOGRAM, METER

def get_h_res(state, eos):
    """
    Calculates residual molar enthalpy for a given state using finite difference of Helmholtz energy.
    """
    T = state.temperature
    V = state.volume
    P = state.pressure
    if callable(P):
        P = P()

    # A_res = sum(contributions)
    conts = state.residual_molar_helmholtz_energy_contributions()
    A_res = conts[0][1]
    for i in range(1, len(conts)):
        A_res += conts[i][1]

    # S_res = -(dA_res/dT)_v
    dT = 0.01 * KELVIN
    molar_density = state.total_moles / V
    state_dt = State(eos, T + dT, density=molar_density, molefracs=state.molefracs)
    conts_dt = state_dt.residual_molar_helmholtz_energy_contributions()
    A_res_dt = conts_dt[0][1]
    for i in range(1, len(conts_dt)):
        A_res_dt += conts_dt[i][1]

    S_res = -(A_res_dt - A_res) / dT
    v_m = 1.0 / molar_density
    P_id = (RGAS * T) / v_m

    # H_res = A_res + T*S_res + (P - P_id)*V_m
    H_res = A_res + T * S_res + (P - P_id) * v_m
    return H_res

def get_calculated_properties(m, sigma, epsilon_k, polymer_data):
    """
    Calculates density (kg/m3) and molar heat capacity (J/mol/K).
    """
    temperatures = polymer_data['T_K'].values
    pressure_bar = polymer_data['P_bar'].unique()[0]
    Mw = polymer_data['Mw'].unique()[0]
    is_pp = "PP" in polymer_data['Type'].unique()[0]
    poly_name = polymer_data['Polymer'].unique()[0]
    M_rep = 42.081 if is_pp else 14.027

    pr = PureRecord(Identifier(name='poly'), Mw, m=m, sigma=sigma, epsilon_k=epsilon_k)
    params_set = Parameters.from_records([pr])
    eos = EquationOfState.pcsaft(params_set)

    rho_calc = []
    cp_calc = []

    for T_val in temperatures:
        T = T_val * KELVIN
        P = pressure_bar * BAR
        state = State(eos, T, pressure=P)
        rho_calc.append(float(state.mass_density() / (KILOGRAM / METER**3)))

        # Finite difference for Cp_res
        dT_fd = 0.1 * KELVIN
        s1 = State(eos, T, pressure=P)
        s2 = State(eos, T + dT_fd, pressure=P)

        h_res1 = get_h_res(s1, eos)
        h_res2 = get_h_res(s2, eos)

        cp_res_total = float((h_res2 - h_res1) / (dT_fd * (JOULE / (MOL * KELVIN))))
        cp_res_rep = cp_res_total / (Mw / M_rep)

        # Improved T-dependent ideal gas heat capacity for PP
        # Using a polynomial fit for PP repeat unit (C3H6)
        if is_pp:
            # Typical polynomial for PP repeat unit: a + b*T + c*T^2
            # Values from literature for liquid-like repeat units
            cp_id_rep = 15.0 + 0.22 * T_val - 1.0e-4 * T_val**2
        else:
            # PE repeat unit (CH2)
            cp_id_rep = -3.38 + 0.0911 * T_val - 3.23e-5 * T_val**2

        cp_calc.append(cp_res_rep + cp_id_rep)

    return np.array(rho_calc), np.array(cp_calc)

def calculate_mape(calc, exp):
    return np.mean(np.abs((calc - exp) / exp)) * 100.0

def estimate_tg_tfusion(m, sigma, epsilon_k, Mw):
    """
    Improved heuristic estimation based on PC-SAFT parameters and Mw.
    For polyolefins:
    """
    # Using more realistic polyolefin values
    # These are still rough but better aligned with PE/PP
    # PE Tg ~ 190-250 K, PP Tg ~ 260-270 K
    # PE Tfusion ~ 400 K, PP Tfusion ~ 440 K
    Tg = 0.8 * epsilon_k
    Tfusion = 1.6 * Tg
    return Tg, Tfusion

def is_self_associating(polymer_name):
    """
    Logic as defined by Ramirez-Velez for polyolefins.
    PE and PP are non-associating (return False).
    """
    associating_polymers = ['PVA', 'PA', 'EVOH']
    return any(p in polymer_name for p in associating_polymers)
