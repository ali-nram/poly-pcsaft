
import numpy as np
import pandas as pd
from feos import Parameters, PureRecord, Identifier, EquationOfState, State, Contributions
from si_units import KELVIN, BAR, JOULE, MOL, KILOGRAM, METER

def get_calculated_properties(m, sigma, epsilon_k, polymer_data, kappa_ab=None, epsilon_k_ab=None, model='pcsaft'):
    """
    Calculates density (kg/m3) and molar heat capacity (J/mol/K) per repeat unit.
    """
    temperatures = polymer_data['T_K'].values
    pressure_bar = polymer_data['P_bar'].unique()[0]
    Mw = polymer_data['Mw'].unique()[0]
    is_pp = "PP" in polymer_data['Type'].unique()[0]
    M_rep = 42.081 if is_pp else 14.027

    # Define model parameters
    model_params = {'m': m, 'sigma': sigma, 'epsilon_k': epsilon_k}
    if kappa_ab is not None and epsilon_k_ab is not None:
        model_params['kappa_ab'] = kappa_ab
        model_params['epsilon_k_ab'] = epsilon_k_ab

    if model == 'pcsaft':
        pr = PureRecord(Identifier(name='poly'), Mw, **model_params)
        params_set = Parameters.from_records([pr])
        eos = EquationOfState.pcsaft(params_set)
    elif model == 'saftvrmie':
        # FeOS Mie parameters: m, sigma, epsilon_k, lr, la (default la=6)
        mie_params = {'m': m, 'sigma': sigma, 'epsilon_k': epsilon_k, 'lr': 12.0, 'la': 6.0}
        pr = PureRecord(Identifier(name='poly'), Mw, **mie_params)
        params_set = Parameters.from_records([pr])
        eos = EquationOfState.saftvrmie(params_set)
    else:
        pr = PureRecord(Identifier(name='poly'), Mw, **model_params)
        params_set = Parameters.from_records([pr])
        eos = EquationOfState.pcsaft(params_set)

    rho_calc = []
    cp_calc = []

    if is_pp:
        A, B, C, D, E = 2.599, 0.28375, -8.06e-5, -1.062e-8, 9.501e-12
    else:
        A, B, C, D, E = -2.071, 0.1444, -1.109e-4, 4.423e-8, -7.051e-12

    for T_val in temperatures:
        T = T_val * KELVIN
        P = pressure_bar * BAR
        try:
            state = State(eos, T, pressure=P)
            rho_calc.append(float(state.mass_density() / (KILOGRAM / METER**3)))
            cp_res_total = float(state.molar_isobaric_heat_capacity(Contributions.Residual) / (JOULE / (MOL * KELVIN)))
            cp_res_rep = cp_res_total / (Mw / M_rep)
            cp_id_rep = A + B*T_val + C*T_val**2 + D*T_val**3 + E*T_val**4
            cp_calc.append(cp_res_rep + cp_id_rep)
        except:
            rho_calc.append(np.nan)
            cp_calc.append(np.nan)

    return np.array(rho_calc), np.array(cp_calc)

def calculate_mape(calc, exp):
    mask = ~np.isnan(calc)
    if not np.any(mask): return 1e6
    return np.mean(np.abs((calc[mask] - exp[mask]) / exp[mask])) * 100.0

def estimate_tg_tfusion(m, sigma, epsilon_k, Mw):
    Tg = 0.8 * epsilon_k
    Tfusion = 1.6 * Tg
    return Tg, Tfusion

def is_self_associating(polymer_name):
    associating_polymers = ['PVA', 'PA', 'EVOH', 'PEO', 'PPO']
    return any(p in polymer_name for p in associating_polymers)
