import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pcsaft_properties import get_calculated_properties

def plot_results():
    best_df = pd.read_csv('best_parameters.csv')
    poly_data_df = pd.read_csv('poly_data.csv')

    # Gross & Sadowski 2001 parameters (doi: 10.1021/ie010449g) for Comparison
    # PE (HDPE/LDPE): m/M = 0.03339 mol/g, sigma = 4.0217 A, epsilon/k = 252.00 K
    # PP (i-PP/s-PP/a-PP): m/M = 0.03152 mol/g, sigma = 4.1473 A, epsilon/k = 298.53 K
    gs_pe = {'m_M': 0.03339, 'sigma': 4.0217, 'epsilon_k': 252.00}
    gs_pp = {'m_M': 0.03152, 'sigma': 4.1473, 'epsilon_k': 298.53}

    # In Gross & Sadowski 2001, for polymers they use m = Mw / M_segment
    # where M_segment is a constant.
    # But often they report m/M. Let's use m = (m/M) * Mw.

    for _, row in best_df.iterrows():
        poly = row['Polymer']
        m = row['m']
        sigma = row['sigma']
        epsilon_k = row['epsilon_k']

        data = poly_data_df[poly_data_df['Polymer'] == poly]
        T = data['T_K'].values
        rho_exp = data['rho_kgm3'].values
        cp_exp = data['Cp_molar_exp'].values
        Mw = data['Mw'].unique()[0]

        # Optimized results
        rho_calc, cp_calc = get_calculated_properties(m, sigma, epsilon_k, data)

        # Gross & Sadowski results for comparison
        is_pp = "PP" in data['Type'].unique()[0]
        gs_params = gs_pp if is_pp else gs_pe
        # Adjust m for GS
        m_gs = gs_params['m_M'] * Mw
        rho_gs, cp_gs = get_calculated_properties(m_gs, gs_params['sigma'], gs_params['epsilon_k'], data)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Density plot
        ax1.plot(T, rho_exp, 'ko', label='Experimental', markersize=4)
        ax1.plot(T, rho_calc, 'r-', label=f'Optimized PC-SAFT (MAPE={row["MAPE_rho"]:.2f}%)')
        ax1.plot(T, rho_gs, 'g--', label=f'Gross & Sadowski 2001 (m/M={gs_params["m_M"]})')
        ax1.set_xlabel('Temperature (K)')
        ax1.set_ylabel('Density (kg/m3)')
        ax1.set_title(f'{poly} Liquid Density')
        ax1.legend()

        # Heat capacity plot
        ax2.plot(T, cp_exp, 'ko', label='Experimental', markersize=4)
        ax2.plot(T, cp_calc, 'b-', label=f'Optimized PC-SAFT (MAPE={row["MAPE_cp"]:.2f}%)')
        ax2.plot(T, cp_gs, 'c--', label='Gross & Sadowski 2001')
        ax2.set_xlabel('Temperature (K)')
        ax2.set_ylabel('Molar Cp (J/mol/K)')
        ax2.set_title(f'{poly} Liquid Heat Capacity')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f'{poly}_comparison_results.png')
        plt.close()

if __name__ == "__main__":
    plot_results()
