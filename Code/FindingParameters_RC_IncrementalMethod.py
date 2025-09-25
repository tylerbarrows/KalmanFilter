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
data_sam1_0degC = pd.concat([df1, df2, df3], ignore_index=True)

# loading SOV vs OCV
# SOC_OCV = np.load('OCV_dOCV_dSOC_SOC_LowCurrent.npz')
# incrementalFlag = 0
SOC_OCV = np.load('OCV_dOCV_dSOC_SOC_Incremental.npz')
incrementalFlag = 1

OCV = SOC_OCV['v1']
dOCV_dSOC = SOC_OCV['v2']
SOC = SOC_OCV['v3']

SOC_interp = interp1d(OCV, SOC)  # finding SOC based on OCV
OCV_interp = interp1d(SOC, OCV)

# current parameters
Ro = 0.112768335
tau1 = 81.70591578
tau2 = 1640.459731
Qnom = 2

def voltage_model(X, R1, R2):
    x1, x2, x3 = X # x1=time, x2 = V(SOC)-i*Ro, x3=i
    return x2 - x3*R1*(1-np.exp(-x1/tau1)) - x3*R2*(1-np.exp(-x1/tau2))

bounds = ([0, 0], [np.inf, np.inf])

def trim_edges(vector, N, flag):
    if flag == 0:
        return vector
    else:
        if N == 0:
            return vector
        elif 2 * N >= len(vector):
            # Avoid returning negative-length slices
            return []
        else:
            return vector[N:-N]

## -----  -----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----  ----
# Sample 1, 0 deg C
data = data_sam1_0degC

# step index and there meanings
discharge_cycle_indices = [1,2,3,4,5,6,7,8,9,10] #cycle index starts at discharge current and ends at end of relaxation period which is when current is off
discharge_current_on_indices = [5,5,5,5,5,5,5,5,5,5]
discharge_current_off_indices = [4,6,6,6,6,6,6,6,6,6]
discharge_prev_cycle_indices = [1,1,2,3,4,5,6,7,8,9]

charge_cycle_indices = [11,12,13,14,15,16,17,18,19]
charge_current_on_indices = [9,9,9,9,9,9,9,9,12]
charge_current_off_indices = [10,10,10,10,10,10,10,10,13]
charge_current_off_prev_indices = [6,10,10,10,10,10,10,10,10]
charge_prev_cycle_indices = [10,11,12,13,14,15,16,17,18,19]

N = 1
discharge_cycle_indices = trim_edges(discharge_cycle_indices,N,incrementalFlag)
discharge_current_on_indices = trim_edges(discharge_current_on_indices,N,incrementalFlag)
discharge_current_off_indices = trim_edges(discharge_current_off_indices,N,incrementalFlag)
discharge_prev_cycle_indices = trim_edges(discharge_prev_cycle_indices,N,incrementalFlag)
charge_cycle_indices = trim_edges(charge_cycle_indices,N,incrementalFlag)
charge_current_on_indices = trim_edges(charge_current_on_indices,N,incrementalFlag)
charge_current_off_indices = trim_edges(charge_current_off_indices,N,incrementalFlag)
charge_current_off_prev_indices = trim_edges(charge_current_off_prev_indices,N,incrementalFlag)
charge_prev_cycle_indices = trim_edges(charge_prev_cycle_indices,N,incrementalFlag)

# Discharge
cycle_ind = 0
for cycle_index in discharge_cycle_indices:
    discharge_current_on = discharge_current_on_indices[cycle_ind]
    discharge_current_off = discharge_current_off_indices[cycle_ind]
    prev_cycle_index = max(1,cycle_index-1)
    data_to_fit = data[(data['Cycle_Index'] == cycle_index) & (data['Step_Index'] == discharge_current_on)]
    data_to_fit_next = data[(data['Cycle_Index'] == cycle_index) & (data['Step_Index'] == discharge_current_off)]
    data_to_fit_prev = data[(data['Cycle_Index'] == prev_cycle_index) & (data['Step_Index'] == discharge_current_off)]
    
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

# Charge
cycle_ind = 0
for cycle_index in charge_cycle_indices:
    charge_current_on = charge_current_on_indices[cycle_ind]
    charge_current_off = charge_current_off_indices[cycle_ind]
    charge_current_off_prev = charge_current_off_prev_indices[cycle_ind]
    prev_cycle_index = cycle_index-1
    data_to_fit = data[(data['Cycle_Index'] == cycle_index) & (data['Step_Index'] == charge_current_on)]
    data_to_fit_next = data[(data['Cycle_Index'] == cycle_index) & (data['Step_Index'] == charge_current_off)]
    data_to_fit_prev = data[(data['Cycle_Index'] == prev_cycle_index) & (data['Step_Index'] == charge_current_off_prev)]
    
    y_data_to_fit = data_to_fit['Voltage(V)'].to_numpy()
    x1_data_to_fit = data_to_fit['Step_Time(s)'].to_numpy()
    x3_data_to_fit = -data_to_fit['Current(A)'].to_numpy()
    dt = x1_data_to_fit[1:] - x1_data_to_fit[:-1]
    OCV_o = data_to_fit_prev['Voltage(V)'].iloc[-1]
    SOC_o = SOC_interp(OCV_o)
    dSOC = - x3_data_to_fit[:-1] * dt / 3600 / Qnom
    SOC = np.empty(len(y_data_to_fit))
    print(len(y_data_to_fit))
    SOC[0] = SOC_o
    for x in range(1,len(SOC)):
        SOC[x] = SOC[x-1] + dSOC[x-1]
    
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
    axs[0].set_title(f'Charge Recovery, Cycle Index: {cycle_index}')
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

    param_track = np.vstack([param_track,solution_found])

    cycle_ind = cycle_ind + 1
    
    # if cycle_index == charge_cycle_indices[-1]:
    #     print('here')
    #     data_to_fit = df[(df['Cycle_Index'] == cycle_index) & (df['Step_Index'] == last_charge_current_off)]
    #     data_to_fit_prev = df[(df['Cycle_Index'] == cycle_index) & (df['Step_Index'] == last_charge_current_on)]
    # else:
    #     data_to_fit = df[(df['Cycle_Index'] == cycle_index) & (df['Step_Index'] == charge_current_off)]
    #     data_to_fit_prev = df[(df['Cycle_Index'] == cycle_index) & (df['Step_Index'] == charge_current_on)]
    
    # y_data_to_fit = data_to_fit['Voltage(V)']
    # x_data_to_fit = data_to_fit['Step_Time(s)']
    # size = len(y_data_to_fit)

    # OCV = data_to_fit['Voltage(V)'].iloc[-1]
    # Vo = data_to_fit_prev['Voltage(V)'].iloc[-1]
    # Io = data_to_fit_prev['Current(A)'].iloc[-1]
    # Ro = (Vo-y_data_to_fit.iloc[0])/Io

    # # for V_threshold in V_threshold_range:
    # x_data_to_fit_prime = x_data_to_fit
    # # y_data_to_fit_prime = (y_data_to_fit-y_data_to_fit.iloc[size-1])*-1
    # y_data_to_fit_prime = (y_data_to_fit-y_data_to_fit.iloc[size-1])

    # initial_guess = [0.25,0.1,225,2100]

    # params, covariance = curve_fit(double_exponential, x_data_to_fit_prime, y_data_to_fit_prime, p0=initial_guess)
    # a1, a2, tau1, tau2 = params
    # y_fit = double_exponential(x_data_to_fit_prime,*params)

    # ss_res = np.sum((y_data_to_fit_prime - y_fit)**2)
    # ss_tot = np.sum((y_data_to_fit_prime - np.mean(y_data_to_fit_prime))**2)
    # r_squared = 1 - (ss_res / ss_tot)

    # solution_found = [OCV, Ro, a1, a2, tau1, tau2, r_squared]
    
    # fig, axs = plt.subplots(2,1,figsize=(8, 5), sharex=True)
    # axs[0].plot(x_data_to_fit_prime,y_data_to_fit_prime,label='Data')
    # axs[0].plot(x_data_to_fit_prime,y_fit,label='Fit')
    # axs[0].set_xlabel('Time (s)')
    # axs[0].set_ylabel('Voltage(V)')
    # axs[0].set_title(f'Discharge Recovery, Cycle Index: {cycle_index}')
    # axs[0].legend()
    
    # axs[1].plot(x_data_to_fit,y_data_to_fit)
    # axs[1].set_xlabel('Time (s)')
    # axs[1].set_ylabel('Voltage(V)')
    # axs[1].set_title(f'Original Data, Cycle Index: {cycle_index}')

    # # if cycle_ind < 1:
    # #     param_track = solution_found
    # # else:
    # #     param_track = np.vstack([param_track,solution_found])
    
    # param_track = np.vstack([param_track,solution_found])
    # cycle_ind = cycle_ind + 1

# print(param_track)
# tau1, tau2, R1, R2, C1, C2, r_squared
df_export = pd.DataFrame(param_track,columns=["tau1","tau2","R1","R2","C1","C2","r2"])
df_export.to_csv('Incremental_OCV_0degC_params_with_RC.csv', index=False)
plt.show()