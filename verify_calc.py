
import numpy as np
import pandas as pd
from pcsaft_properties import get_calculated_properties

def test_calculator():
    # Test with a-PP parameters from results
    m = 873.35
    sigma = 5.5
    epsilon_k = 237.43

    dummy_data = pd.DataFrame({
        'T_K': [300.0],
        'P_bar': [1.0],
        'Mw': [100000.0], # Molar mass
        'Type': ['a-PP']
    })

    rho, cp = get_calculated_properties(m, sigma, epsilon_k, dummy_data)

    print(f"Test Result for a-PP at 300K:")
    print(f"Density: {rho[0]:.2f} kg/m3")
    print(f"Heat Capacity: {cp[0]:.2f} J/mol/K")

    assert rho[0] > 0
    assert cp[0] > 0
    print("Verification successful!")

if __name__ == "__main__":
    test_calculator()
