##
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy.optimize import curve_fit

# Incremental Current OCV - Sample 1 - Data for 0 deg C
df1 = pd.read_excel('../Data/02_26_2016_SP20-1_0C_incrementalOCV.xls',sheet_name='Channel_1-005_1')
df2 = pd.read_excel('../Data/02_26_2016_SP20-1_0C_incrementalOCV.xls',sheet_name='Channel_1-005_2')
df3 = pd.read_excel('../Data/02_26_2016_SP20-1_0C_incrementalOCV.xls',sheet_name='Channel_1-005_3')
df = pd.concat([df1, df2, df3], ignore_index=True)

# step index and there meanings
discharge_current_on = 5
discharge_current_off = 6
cycle_indices = [1,2,3,4,5,6,7,8,9,10] #cycle index starts at discharge current and ends at end of relaxation period which is when current is off

def double_exponential(t, a1, a2, tau1, tau2):
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)

cycle_ind = 0
for cycle_index in cycle_indices:
    data_to_fit = df[(df['Cycle_Index'] == cycle_index) & (df['Step_Index'] == discharge_current_off)]
    y_data_to_fit = data_to_fit['Voltage(V)']
    x_data_to_fit = data_to_fit['Step_Time(s)']
    size = len(y_data_to_fit)

    V_threshold_range = np.arange(0.7,0.1,-0.005)

    R2_ticker = 0
    ind = 0

    for V_threshold in V_threshold_range:
        x_data_to_fit_prime = x_data_to_fit
        y_data_to_fit_prime = (y_data_to_fit-y_data_to_fit.iloc[size-1])*-1

        start_index = y_data_to_fit_prime[y_data_to_fit_prime < V_threshold].index.min()
        x_data_to_fit_prime = x_data_to_fit_prime.loc[start_index:].reset_index(drop=True)
        y_data_to_fit_prime = y_data_to_fit_prime.loc[start_index:].reset_index(drop=True)
        x_data_to_fit_prime = x_data_to_fit_prime - x_data_to_fit_prime.iloc[0]

        initial_guess = [0.25,0.1,225,2100]
        if ind > 0:
            initial_guess = params

        params, covariance = curve_fit(double_exponential, x_data_to_fit_prime, y_data_to_fit_prime, p0=initial_guess)
        a1, a2, tau1, tau2 = params
        y_fit = double_exponential(x_data_to_fit_prime,*params)

        ss_res = np.sum((y_data_to_fit_prime - y_fit)**2)
        ss_tot = np.sum((y_data_to_fit_prime - np.mean(y_data_to_fit_prime))**2)
        r_squared = 1 - (ss_res / ss_tot)

        if r_squared > R2_ticker:
            V_threshold_use = V_threshold
            R2_ticker = r_squared
            guess_to_use = [(y_data_to_fit.iloc[0] - V_threshold_use)/1, a1, a2, tau1, tau2, r_squared]

        ind = ind  + 1
    
    fig, axs = plt.subplots(2,1,figsize=(8, 5), sharex=True)
    axs[0].plot(x_data_to_fit_prime,y_data_to_fit_prime,label='Data')
    axs[0].plot(x_data_to_fit_prime,y_fit,label='Fit')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Voltage(V)')
    axs[0].set_title(f'Discharge Recovery, Cycle Index: {cycle_index}')
    axs[0].legend()
    
    axs[1].plot(x_data_to_fit,y_data_to_fit)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Voltage(V)')
    axs[1].set_title(f'Original Data, Cycle Index: {cycle_index}')

    if cycle_ind < 1:
        param_track = guess_to_use
    else:
        param_track = np.vstack([param_track,guess_to_use])
    cycle_ind = cycle_ind + 1

print(param_track)
plt.show()