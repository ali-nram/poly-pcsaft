import numpy as np
import pandas as pd
from scipy.optimize import minimize
from feos import Parameters, PureRecord, Identifier, EquationOfState, State, Contributions
from si_units import KELVIN, BAR, JOULE, MOL, KILOGRAM, METER
from pcsaft_properties import is_self_associating

def objective_function(params, polymer_data, weight_rho, weight_cp):
    """
    params: [m, sigma, epsilon_k, (kappa_assoc, epsilon_assoc_k if associating)]
    """
    poly_name = polymer_data['Polymer'].unique()[0]
    is_assoc = is_self_associating(poly_name)

    # Extract data
    temperatures = polymer_data['T_K'].values
    pressure_bar = polymer_data['P_bar'].unique()[0]
    rho_exp = polymer_data['rho_kgm3'].values
    Mw = polymer_data['Mw'].unique()[0]

    try:
        from pcsaft_properties import get_calculated_properties
        rho_calc, cp_calc = get_calculated_properties(params[0], params[1], params[2], polymer_data)

        # User requested: "average square deviations between experimental and calculated properties"
        OF_rho = np.mean((rho_calc - rho_exp)**2)

        if weight_cp > 0:
            cp_exp = polymer_data['Cp_molar_exp'].values
            OF_cp = np.mean((cp_calc - cp_exp)**2)
        else:
            OF_cp = 0.0

        return weight_rho * OF_rho + weight_cp * OF_cp

    except Exception as e:
        return 1e10

def optimize_polymer(polymer_data, weight_rho=1.0, weight_cp=1.0):
    poly_name = polymer_data['Polymer'].unique()[0]
    Mw = polymer_data['Mw'].unique()[0]
    is_assoc = is_self_associating(poly_name)

    # Improved initial guesses for polymers
    m0 = 0.026 * Mw
    sigma0 = 4.0
    epsilon_k0 = 250.0
    initial_guess = [m0, sigma0, epsilon_k0]
    bounds = [(m0 * 0.1, m0 * 10.0), (3.0, 5.5), (100.0, 500.0)]

    if is_assoc:
        initial_guess += [0.01, 1000.0]
        bounds += [(0.0, 0.1), (0.0, 5000.0)]

    res = minimize(objective_function, initial_guess, args=(polymer_data, weight_rho, weight_cp),
                   bounds=bounds, method='L-BFGS-B')
    return res
