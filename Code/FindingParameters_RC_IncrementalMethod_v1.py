##
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# Incremental Current OCV - Sample 1 - Data for 0 deg C
df1 = pd.read_excel('../Data/02_26_2016_SP20-1_0C_incrementalOCV.xls',sheet_name='Channel_1-005_1')
df2 = pd.read_excel('../Data/02_26_2016_SP20-1_0C_incrementalOCV.xls',sheet_name='Channel_1-005_2')
df3 = pd.read_excel('../Data/02_26_2016_SP20-1_0C_incrementalOCV.xls',sheet_name='Channel_1-005_3')
df = pd.concat([df1, df2, df3], ignore_index=True)

# plot all data
# fig, axs = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
fig, axs = plt.subplots(4, 1, figsize=(8, 5), sharex=True)
size_for_markers_plot = 2
size_for_markers_scatter = 22

axs[0].plot(df['Test_Time(s)'], df['Voltage(V)'], marker='o', markersize=size_for_markers_plot)
axs[0].set_title(f'All data')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Voltage (V)')
axs[0].grid(True)

axs[1].plot(df['Test_Time(s)'], df['dV/dt(V/s)'], marker='o', markersize=size_for_markers_plot)
axs[1].set_xlabel('Time')
axs[1].set_ylabel('dV/dt (V/s)')
axs[1].grid(True)

axs[2].plot(df['Test_Time(s)'], df['Current(A)'], marker='o', markersize=size_for_markers_plot)
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Current (A)')
axs[2].grid(True)

axs[3].scatter(df['Test_Time(s)'], df['Step_Index'], label='Step Index', marker='o', s=size_for_markers_scatter)
axs[3].scatter(df['Test_Time(s)'], df['Cycle_Index'], label='Cycle Index', marker='o', s=size_for_markers_scatter)
y_max = max(max(df['Step_Index']),max(df['Cycle_Index']))
axs[3].set_yticks(range(0, int(y_max)+1))
axs[3].set_xlabel('Time')
axs[3].set_ylabel('Index')
axs[3].legend()
axs[3].grid(True)

# loading SOV vs OCV
SOC_OCV = np.load('OVC_dOCV_dSOV_SOC_LowCurrent.npz')
# SOC_OCV = np.load('OVC_dOCV_dSOV_SOC_Incremental.npz')

OCV = SOC_OCV['v1']
dOCV_dSOC = SOC_OCV['v2']
SOC = SOC_OCV['v3']

# plot this too
fig, axs = plt.subplots(figsize=(8, 5), sharex=True)
axs.plot(SOC, OCV)
axs.set_title('OCV vs SOC')
axs.set_xlabel('SOC')
axs.set_ylabel('OCV (V)')
axs.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
axs.grid(True)

fig, axs = plt.subplots(figsize=(8, 5), sharex=True)
axs.plot(SOC, OCV)
axs.set_title('OCV vs SOC')
axs.set_xlabel('SOC')
axs.set_ylabel('OCV (V)')
axs.set_xlim([0.1, 0.9])
axs.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
axs.grid(True)

fig, axs = plt.subplots(figsize=(8, 5), sharex=True)
axs.plot(OCV, SOC)
axs.set_title('SOC vs OCV')
axs.set_xlabel('OCV (V)')
axs.set_ylabel('SOC')
axs.grid(True)

SOC_interp = interp1d(OCV, SOC)  # finding SOC based on OCV
OCV_interp = interp1d(SOC, OCV)

# current parameters
Ro = 0.112768335
tau1 = 81.70591578
tau2 = 1640.459731
Qnom = 2

# step index and there meanings
discharge_current_on = 5
discharge_current_off = 6
discharge_cycle_indices = [1,2,3,4,5,6,7,8,9,10] #cycle index starts at discharge current and ends at end of relaxation period which is when current is off
first_discharge_current_off = 4

charge_current_on = 9
charge_current_off = 10
charge_cycle_indices = [11,12,13,14,15,16,17,18,19]
last_charge_current_on = 12
last_charge_current_off = 13

# def double_exponential(t, a1, a2, tau1, tau2):
#     return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)

# def voltage_model(X, a1, a2, R1, R2):
#     x1, x2, x3 = X # x1=time, x2 = V(SOC)-i*Ro, x3=i
#     return x2 - a1*np.exp(-x1/tau1) + x3*R1*(1-np.exp(-x1/tau1)) - a2*np.exp(-x1/tau2) + x3*R2*(1-np.exp(-x1/tau2))

def voltage_model(X, R1, R2):
    x1, x2, x3 = X # x1=time, x2 = V(SOC)-i*Ro, x3=i
    return x2 - x3*R1*(1-np.exp(-x1/tau1)) - x3*R2*(1-np.exp(-x1/tau2))

bounds = ([0, 0], [np.inf, np.inf])

# Discharge
cycle_ind = 0
for cycle_index in discharge_cycle_indices:
    data_to_fit = df[(df['Cycle_Index'] == cycle_index) & (df['Step_Index'] == discharge_current_on)]
    data_to_fit_next = df[(df['Cycle_Index'] == cycle_index) & (df['Step_Index'] == discharge_current_off)]
    if cycle_index == discharge_cycle_indices[0]:
        data_to_fit_prev = df[(df['Cycle_Index'] == cycle_index) & (df['Step_Index'] == first_discharge_current_off)]
    else:
        data_to_fit_prev = df[(df['Cycle_Index'] == cycle_index-1) & (df['Step_Index'] == discharge_current_off)]
    
    y_data_to_fit = data_to_fit['Voltage(V)'].to_numpy()
    x1_data_to_fit = data_to_fit['Step_Time(s)'].to_numpy()
    x3_data_to_fit = -data_to_fit['Current(A)'].to_numpy()
    dt = x1_data_to_fit[1:] - x1_data_to_fit[:-1]
    OCV_o = data_to_fit_prev['Voltage(V)'].iloc[-1]
    SOC_o = SOC_interp(OCV_o)
    dSOC = - x3_data_to_fit[:-1] * dt / 3600 / Qnom
    SOC = np.empty(len(y_data_to_fit))
    SOC[0] = SOC_o
    for x in range(1,len(SOC)):
        SOC[x] = SOC[x-1] + dSOC[x-1]

    # SOC = np.concatenate(([SOC_o], dSOC)) # use below of this if this doesn't work
    # SOC = pd.concat([pd.Series([SOC_o]), dSOC], ignore_index=True)
    OCV = OCV_interp(SOC)
    x2_data_to_fit = OCV - x3_data_to_fit*Ro

    X = np.vstack((x1_data_to_fit, x2_data_to_fit, x3_data_to_fit))
    initial_guess = [10,100]
    params, covariance = curve_fit(voltage_model, X, y_data_to_fit, p0=initial_guess, bounds=bounds) # this where I am

    R1, R2 = params
    C1 = tau1/R1
    C2 = tau2/R2

    y_fit = voltage_model(X,*params)

    ss_res = np.sum((y_data_to_fit - y_fit)**2)
    ss_tot = np.sum((y_data_to_fit - np.mean(y_data_to_fit))**2)
    r_squared = 1 - (ss_res / ss_tot)

    solution_found = [tau1, tau2, R1, R2, C1, C2, r_squared]
    
    fig, axs = plt.subplots(5,1,figsize=(8, 5), sharex=True)
    axs[0].plot(x1_data_to_fit,y_data_to_fit,label='Data')
    axs[0].plot(x1_data_to_fit,y_fit,label='Fit')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Voltage(V)')
    axs[0].set_title(f'Discharge Recovery, Cycle Index: {cycle_index}')
    axs[0].legend()
    
    axs[1].plot(x1_data_to_fit,y_data_to_fit)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Voltage(V)')
    # axs[1].set_title(f'Original Data, Cycle Index: {cycle_index}')

    axs[2].plot(x1_data_to_fit,SOC)
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('SOC')
    # axs[2].set_title(f'SOC: {cycle_index}')

    axs[3].plot(x1_data_to_fit,OCV)
    axs[3].set_xlabel('Time (s)')
    axs[3].set_ylabel('OCV (V)')
    # axs[3].set_title(f'OCV (V): {cycle_index}')

    axs[4].plot(x1_data_to_fit,x2_data_to_fit)
    axs[4].set_xlabel('Time (s)')
    axs[4].set_ylabel('$V_{OC}$-$V_{o}$ (V)')
    # axs[4].set_title(f'$V_{{OC}}$-$V_{{o}}$ (V): {cycle_index}')

    if cycle_ind < 1:
        param_track = solution_found
    else:
        param_track = np.vstack([param_track,solution_found])

    cycle_ind = cycle_ind + 1

# plt.show()

# # Charge
# cycle_ind = 0
# for cycle_index in charge_cycle_indices:
#     print(cycle_index)
#     if cycle_index == charge_cycle_indices[-1]:
#         print('here')
#         data_to_fit = df[(df['Cycle_Index'] == cycle_index) & (df['Step_Index'] == last_charge_current_off)]
#         data_to_fit_prev = df[(df['Cycle_Index'] == cycle_index) & (df['Step_Index'] == last_charge_current_on)]
#     else:
#         data_to_fit = df[(df['Cycle_Index'] == cycle_index) & (df['Step_Index'] == charge_current_off)]
#         data_to_fit_prev = df[(df['Cycle_Index'] == cycle_index) & (df['Step_Index'] == charge_current_on)]
    
#     y_data_to_fit = data_to_fit['Voltage(V)']
#     x_data_to_fit = data_to_fit['Step_Time(s)']
#     size = len(y_data_to_fit)

#     OCV = data_to_fit['Voltage(V)'].iloc[-1]
#     Vo = data_to_fit_prev['Voltage(V)'].iloc[-1]
#     Io = data_to_fit_prev['Current(A)'].iloc[-1]
#     Ro = (Vo-y_data_to_fit.iloc[0])/Io

#     # for V_threshold in V_threshold_range:
#     x_data_to_fit_prime = x_data_to_fit
#     # y_data_to_fit_prime = (y_data_to_fit-y_data_to_fit.iloc[size-1])*-1
#     y_data_to_fit_prime = (y_data_to_fit-y_data_to_fit.iloc[size-1])

#     initial_guess = [0.25,0.1,225,2100]

#     params, covariance = curve_fit(double_exponential, x_data_to_fit_prime, y_data_to_fit_prime, p0=initial_guess)
#     a1, a2, tau1, tau2 = params
#     y_fit = double_exponential(x_data_to_fit_prime,*params)

#     ss_res = np.sum((y_data_to_fit_prime - y_fit)**2)
#     ss_tot = np.sum((y_data_to_fit_prime - np.mean(y_data_to_fit_prime))**2)
#     r_squared = 1 - (ss_res / ss_tot)

#     solution_found = [OCV, Ro, a1, a2, tau1, tau2, r_squared]
    
#     fig, axs = plt.subplots(2,1,figsize=(8, 5), sharex=True)
#     axs[0].plot(x_data_to_fit_prime,y_data_to_fit_prime,label='Data')
#     axs[0].plot(x_data_to_fit_prime,y_fit,label='Fit')
#     axs[0].set_xlabel('Time (s)')
#     axs[0].set_ylabel('Voltage(V)')
#     axs[0].set_title(f'Discharge Recovery, Cycle Index: {cycle_index}')
#     axs[0].legend()
    
#     axs[1].plot(x_data_to_fit,y_data_to_fit)
#     axs[1].set_xlabel('Time (s)')
#     axs[1].set_ylabel('Voltage(V)')
#     axs[1].set_title(f'Original Data, Cycle Index: {cycle_index}')

#     # if cycle_ind < 1:
#     #     param_track = solution_found
#     # else:
#     #     param_track = np.vstack([param_track,solution_found])
    
#     param_track = np.vstack([param_track,solution_found])
#     cycle_ind = cycle_ind + 1

# print(param_track)
# tau1, tau2, R1, R2, C1, C2, r_squared
df_export = pd.DataFrame(param_track,columns=["tau1","tau2","R1","R2","C1","C2","r2"])
df_export.to_csv('Incremental_OCV_0degC_params_with_RC.csv', index=False)
plt.show()