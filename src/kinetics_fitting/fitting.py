# import packages
from pathlib import Path
from typing import Union, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit #for second order reaction rates.

def alt_five_param_logistic_equation(x, A1, A2, c, p, S):
    return A1 + ((A2 - A1) / (1 + (x/c)**p)**S) #this is the original equation


def fit_reaction_segments(df: pd.DataFrame, wavelength: Union[int, str], A0_guess: float, repnumber: str, result_folder: Path) -> Optional[dict]:
    data = df[df['Wavelength'] == wavelength].copy()
    segment1 = data
    segment1['Time_normalized'] = segment1['Time'] - segment1['Time'].min() # Normalize time to start at 0 for each segment
    
    wavelength = str(wavelength)
    if wavelength == '540':
        fit_func = alt_five_param_logistic_equation
    else:
        fit_func = alt_five_param_logistic_equation
        
    #print which function is being used
    print(f'Fitting data for {wavelength} nm')
        
    # Set better initial guesses (this is the offset, C)
    final_intensity = segment1['Intensity'].iloc[-1]
    
    try:
        popt, pcov = curve_fit(
            fit_func, 
            segment1['Time_normalized'], #x
            segment1['Intensity'], #y
            p0=[final_intensity, A0_guess, final_intensity/1.5, 1e-3, 1.0],  # initial guesses for A, A2, x0, p, S 
            #NOTE: we don't divide final_intensity by 2 because the data is a 5PL so we expect the inflection point to be above or below, not in the exact middle.
            bounds=([0, 0, 0, 1e-8, 0],[np.inf, np.inf, np.inf, np.inf, np.inf]),
            maxfev=10000)
        
    except RuntimeError as e:
        print(f"Failed to fit data for {wavelength}nm: {str(e)}")
        return None
    
    # if A0_guess is your initial conc. (in same units as Intensity)
    A1, A2, c, p, S = popt

    # note that for second order kinetics of A + A -> P, the rate law is:
    # t_1/2 = 1 / k[A]_0 so we can simply take the fitted value of c and divide by the initial concentration to get k.
    k_effective = 1.0 / (c * A0_guess)
    
    if repnumber.startswith('1'):
        k_effective /= 1e-0
    elif repnumber.startswith('2'):
        k_effective /= 2e-0
    elif repnumber.startswith('5'):
        k_effective /= 5e-0
    elif repnumber.startswith('8'):
        k_effective /= 8e-0
    else:
        print(f"Unknown replicate number: {repnumber}. Using k_per_s as is.")

    k_effective = 1/k_effective

    print(f"Derived 2nd‐order k = {k_effective:.3e} (same units as your 2nd‐order fit)")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.scatter(data['Time'], data['Intensity'], alpha=0.5, label='Data')
    
    if popt is not None:
        t_fit = np.linspace(0, segment1['Time_normalized'].max(), 100)
        y_fit = fit_func(t_fit, *popt)
        plt.plot(t_fit + segment1['Time'].min(), y_fit, 'r-', label=f'Fit 1 (k={k_effective:.3e} s⁻¹)')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity (a.u.)')
    plt.title(f'Second Order Reaction Fits for {wavelength}nm, {repnumber}')
    plt.legend()

    popt[0] = k_effective
    
    #save figure as png
    plt.savefig(result_folder / f'2ndfit_{wavelength}_{repnumber}.png', dpi=300, bbox_inches='tight')
    
    #save popt and pcov parameters to a text file as well
    with open(result_folder / f'2ndfit_{wavelength}_r{repnumber}.txt', 'w') as f:
        f.write(f'params: {popt} \n covariance: {pcov}')
        
    return {
        'params': popt,
        'covariance': pcov,
        'A0_guess' : A0_guess
    }


def compute_inflection(A1, A2, c, p, S):
    """
    Given 5PL parameters for a sigmoidal curve defined as:
    
       f(x) = A1 + (A2 - A1)/(1 + (x/c)**p)**S,
    
    this function computes the inflection point (x at maximum derivative)
    and the maximum derivative f'(x_inf).
    
    The inflection point is given by the second derivative to find the maxima of f'(x). the inflection point ocurs when f''(x)=0, and we assume that p>1.
       x_inflect = c*((p-1)/(1+S*p))**(1/p)
    and the derivative is given by:
       f'(x) = (-S*p(a-d))/c**p) * x**(p-1) * (1 + (x/c)**p)**(-S-1)
       
    If p <= 1, the formula is not valid because p-1 where p=1 is 0 and x^0 = 1 hence all x would cancel out and the equation would no longer be solvable for x.
    """
    if p <= 1:
        return np.nan, np.nan  # or handle differently if needed
    
    u = (p - 1) / (p * S + 1)
    x_inf = c * (u)**(1/p)
    ratio = x_inf / c
    derivative = ((-S * p *  (A2 - A1)) / c**p) * ratio**(p-1) *  ((1 + (ratio/c)**p)**(-S - 1))
    return x_inf, derivative