import numpy as np
import pandas as pd
from scipy.optimize import minimize
from feos import Parameters, PureRecord, Identifier, EquationOfState, State
# from si_units import *
from si_units import KELVIN, BAR, JOULE, MOL, KILOGRAM, METER
from pcsaft_properties import get_h_res, is_self_associating

def objective_function(params, polymer_data, weight_rho, weight_cp):
    """
    params: [m, sigma, epsilon_k, (kappa_assoc, epsilon_assoc_k if associating)]
    polymer_data: DataFrame with experimental T, P, rho, Cp_molar
    """
    poly_name = polymer_data['Polymer'].unique()[0]
    is_assoc = is_self_associating(poly_name)

    # Extract data
    temperatures = polymer_data['T_K'].values
    pressure_bar = polymer_data['P_bar'].unique()[0]
    rho_exp = polymer_data['rho_kgm3'].values
    Mw = polymer_data['Mw'].unique()[0]
    is_pp = "PP" in polymer_data['Type'].unique()[0]
    M_rep = 42.081 if is_pp else 14.027

    try:
        if is_assoc:
            m, sigma, epsilon_k, kappa_assoc, epsilon_assoc_k = params
            pr = PureRecord(
                identifier=Identifier(name='poly'),
                molarweight=Mw,
                model_record={
                    'm': m,
                    'sigma': sigma,
                    'epsilon_k': epsilon_k,
                    'kappa_ab': kappa_assoc,
                    'epsilon_k_ab': epsilon_assoc_k
                },
                association_sites=["A", "B"]
            )
            # Note: The actual feos API for association sites might require
            # building an AssociationRecord. Given PE/PP are non-assoc,
            # we focus on the 3-parameter case.
        else:
            m, sigma, epsilon_k = params
            pr = PureRecord(Identifier(name='poly'), Mw, m=m, sigma=sigma, epsilon_k=epsilon_k)

        params_set = Parameters.from_records([pr])
        eos = EquationOfState.pcsaft(params_set)

        # Calculate Density
        rho_calc = []
        for T_val in temperatures:
            T = T_val * KELVIN
            P = pressure_bar * BAR
            state = State(eos, T, pressure=P)
            rho_calc.append(float(state.mass_density() / (KILOGRAM / METER**3)))
        rho_calc = np.array(rho_calc)
        OF_rho = np.mean(((rho_calc - rho_exp) / rho_exp)**2)

        # Calculate Heat Capacity if weight_cp > 0
        if weight_cp > 0:
            cp_exp = polymer_data['Cp_molar_exp'].values
            cp_calc = []
            for T_val in temperatures:
                T = T_val * KELVIN
                P = pressure_bar * BAR
                state = State(eos, T, pressure=P)
                dT_fd = 0.1 * KELVIN
                s1 = State(eos, T, pressure=P)
                s2 = State(eos, T + dT_fd, pressure=P)
                h_res1 = get_h_res(s1, eos)
                h_res2 = get_h_res(s2, eos)
                cp_res_total = float((h_res2 - h_res1) / (dT_fd * (JOULE / (MOL * KELVIN))))
                cp_res_rep = cp_res_total / (Mw / M_rep)

                if is_pp:
                    # Improved T-dependent ideal gas heat capacity for PP
                    cp_id_rep = 15.0 + 0.22 * T_val - 1.0e-4 * T_val**2
                else:
                    cp_id_rep = -3.38 + 0.0911 * T_val - 3.23e-5 * T_val**2

                cp_calc.append(cp_res_rep + cp_id_rep)
            cp_calc = np.array(cp_calc)
            OF_cp = np.mean(((cp_calc - cp_exp) / cp_exp)**2)
        else:
            OF_cp = 0.0

        return weight_rho * OF_rho + weight_cp * OF_cp

    except Exception as e:
        return 1e10

def optimize_polymer(polymer_data, weight_rho=1.0, weight_cp=1.0):
    poly_name = polymer_data['Polymer'].unique()[0]
    Mw = polymer_data['Mw'].unique()[0]
    is_assoc = is_self_associating(poly_name)

    m0 = 0.033 * Mw
    sigma0 = 4.0
    epsilon_k0 = 250.0
    initial_guess = [m0, sigma0, epsilon_k0]
    bounds = [(m0 * 0.1, m0 * 10.0), (3.0, 5.5), (100.0, 500.0)]

    if is_assoc:
        initial_guess += [0.01, 1000.0] # kappa_assoc, epsilon_assoc_k
        bounds += [(0.0, 0.1), (0.0, 5000.0)]

    res = minimize(objective_function, initial_guess, args=(polymer_data, weight_rho, weight_cp),
                   bounds=bounds, method='L-BFGS-B')
    return res
