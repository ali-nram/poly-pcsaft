import pandas as pd
import numpy as np
from optimizer import optimize_polymer
from pcsaft_properties import get_calculated_properties, calculate_mape, estimate_tg_tfusion

def run_weight_study(w_rho=2.0, w_cp_list=None):
    if w_cp_list is None:
        w_cp_list = np.arange(0, 1.51, 0.15)

    df = pd.read_csv('poly_data.csv')
    polymers = df['Polymer'].unique()

    all_results = []

    for poly in polymers:
        print(f"Study for {poly}...")
        poly_data = df[df['Polymer'] == poly]
        Mw = poly_data['Mw'].unique()[0]

        for w_cp in w_cp_list:
            print(f"  w_rho={w_rho}, w_cp={w_cp:.2f}...")
            res = optimize_polymer(poly_data, weight_rho=w_rho, weight_cp=w_cp)

            if res.success:
                m, sigma, epsilon_k = res.x
                rho_calc, cp_calc = get_calculated_properties(m, sigma, epsilon_k, poly_data)

                mape_rho = calculate_mape(rho_calc, poly_data['rho_kgm3'].values)
                mape_cp = calculate_mape(cp_calc, poly_data['Cp_molar_exp'].values)
                mape_total = mape_rho + mape_cp

                Tg, Tfusion = estimate_tg_tfusion(m, sigma, epsilon_k, Mw)

                all_results.append({
                    'Polymer': poly,
                    'w_rho': w_rho,
                    'w_cp': w_cp,
                    'm': m,
                    'sigma': sigma,
                    'epsilon_k': epsilon_k,
                    'MAPE_rho': mape_rho,
                    'MAPE_cp': mape_cp,
                    'MAPE_total': mape_total,
                    'Tg': Tg,
                    'Tfusion': Tfusion
                })

    return pd.DataFrame(all_results)

if __name__ == "__main__":
    results_df = run_weight_study()
    results_df.to_csv('weight_study_results.csv', index=False)

    # Hierarchical Selection Ladder: Density > Heat Capacity > Total MAPE
    # Criteria:
    # 1. MAPE_rho < 0.5% (Pseudo-experimental limit)
    # 2. Minimum MAPE_cp
    # 3. Minimum MAPE_total

    best_results = []
    for poly in results_df['Polymer'].unique():
        poly_res = results_df[results_df['Polymer'] == poly].copy()

        # Priority 1: Density MAPE < 0.5%
        qualified = poly_res[poly_res['MAPE_rho'] < 0.5]

        if qualified.empty:
            # If none meet 0.5%, take the top 3 with lowest MAPE_rho
            qualified = poly_res.nsmallest(3, 'MAPE_rho')

        # Priority 2: From qualified, pick lowest MAPE_cp
        best = qualified.loc[qualified['MAPE_cp'].idxmin()]
        best_results.append(best)

    best_df = pd.DataFrame(best_results)
    print("\nBest Parameters per Polymer:")
    print(best_df[['Polymer', 'w_cp', 'MAPE_rho', 'MAPE_cp', 'MAPE_total']])
    best_df.to_csv('best_parameters.csv', index=False)
