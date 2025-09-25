##
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy.optimize import curve_fit

# Incremental Current OCV - Sample 1 - Data for 0 deg C
# samNum = 1
# T = 0
# df1 = pd.read_excel('../Data/02_26_2016_SP20-1_0C_incrementalOCV.xls',sheet_name='Channel_1-005_1')
# df2 = pd.read_excel('../Data/02_26_2016_SP20-1_0C_incrementalOCV.xls',sheet_name='Channel_1-005_2')
# df3 = pd.read_excel('../Data/02_26_2016_SP20-1_0C_incrementalOCV.xls',sheet_name='Channel_1-005_3')
# df = pd.concat([df1, df2, df3], ignore_index=True)
# # step index and there meanings
# # discharge
# discharge_current_on_indices = [5,5,5,5,5,5,5,5,5,5]
# discharge_current_off_indices = [6,6,6,6,6,6,6,6,6,6]
# discharge_cycle_indices = [1,2,3,4,5,6,7,8,9,10] 
# # charge
# charge_current_on_indices = [9,9,9,9,9,9,9,9,12]
# charge_current_off_indices = [10,10,10,10,10,10,10,10,13]
# charge_cycle_indices = [11,12,13,14,15,16,17,18,19]

# Incremental Current OCV - Sample 1 - Data for 25 deg C
samNum = 1
T = 25
df1 = pd.read_excel('../Data/12_2_2015_Incremental OCV test_SP20-1.xlsx',sheet_name='Channel_1-005_1')
df2 = pd.read_excel('../Data/12_2_2015_Incremental OCV test_SP20-1.xlsx',sheet_name='Channel_1-005_2')
df3 = pd.read_excel('../Data/12_2_2015_Incremental OCV test_SP20-1.xlsx',sheet_name='Channel_1-005_3')
df = pd.concat([df1, df2, df3], ignore_index=True)
# step index and there meanings
# discharge
discharge_current_on_indices = [5,5,5,5,5,5,5,5,5,5]
discharge_current_off_indices = [6,6,6,6,6,6,6,6,6,8]
discharge_cycle_indices = [1,2,3,4,5,6,7,8,9,10] 
# charge
charge_current_on_indices = [9,9,9,9,9,9,9]
charge_current_off_indices = [10,10,10,10,10,10,10]
charge_cycle_indices = [11,12,13,14,15,16,17]

# Incremental Current OCV - Sample 1 - Data for 45 deg C
# samNum = 1
# T = 45
# df1 = pd.read_excel('../Data/12_09_2015_Incremental OCV test_SP20-1.xlsx',sheet_name='Channel_1-005_1')
# df2 = pd.read_excel('../Data/12_09_2015_Incremental OCV test_SP20-1.xlsx',sheet_name='Channel_1-005_2')
# df3 = pd.read_excel('../Data/12_09_2015_Incremental OCV test_SP20-1.xlsx',sheet_name='Channel_1-005_3')
# df = pd.concat([df1, df2, df3], ignore_index=True)
# # step index and there meanings
# # discharge
# discharge_current_on_indices = [5,5,5,5,5,5,5,5,5,5]
# discharge_current_off_indices = [6,6,6,6,6,6,6,6,6,6]
# discharge_cycle_indices = [1,2,3,4,5,6,7,8,9,10] 
# # discharge previous (used for new Ro calculation)
# discharge_current_on_prev_indices = [5,5,5,5,5,5,5,5,5,5]
# discharge_current_off_prev_indices = [4,6,6,6,6,6,6,6,6,6]
# discharge_cycle_prev_indices = [1,1,2,3,4,5,6,7,8,9] 
# # charge
# charge_current_on_indices = [9,9,9,9,9,9,9,9,12]
# charge_current_off_indices = [10,10,10,10,10,10,10,10,13]
# charge_cycle_indices = [11,12,13,14,15,16,17,18,19]

# Incremental Current OCV - Sample 2 - Data for 0 deg C
# samNum = 2
# T = 0
# df1 = pd.read_excel('../Data/03_09_2016_SP20-3_0C_incrementalOCV.xls',sheet_name='Channel_1-005_1')
# df2 = pd.read_excel('../Data/03_09_2016_SP20-3_0C_incrementalOCV.xls',sheet_name='Channel_1-005_2')
# df3 = pd.read_excel('../Data/03_09_2016_SP20-3_0C_incrementalOCV.xls',sheet_name='Channel_1-005_3')
# df = pd.concat([df1, df2, df3], ignore_index=True)
# # step index and there meanings
# # discharge
# discharge_current_on_indices = [5,5,5,5,5,5,5,5,5,5]
# discharge_current_off_indices = [6,6,6,6,6,6,6,6,6,8]
# discharge_cycle_indices = [1,2,3,4,5,6,7,8,9,10] 
# # charge
# charge_current_on_indices = [9,9,9,9,9,9,9,9]
# charge_current_off_indices = [10,10,10,10,10,10,10,13]
# charge_cycle_indices = [11,12,13,14,15,16,17,18]

# Incremental Current OCV - Sample 2 - Data for 25 deg C
# samNum = 2
# T = 25
# df1 = pd.read_excel('../Data/12_2_2015_Incremental OCV test_SP20-3.xlsx',sheet_name='Channel_1-006_1')
# df2 = pd.read_excel('../Data/12_2_2015_Incremental OCV test_SP20-3.xlsx',sheet_name='Channel_1-006_2')
# df3 = pd.read_excel('../Data/12_2_2015_Incremental OCV test_SP20-3.xlsx',sheet_name='Channel_1-006_3')
# df = pd.concat([df1, df2, df3], ignore_index=True)
# # step index and there meanings
# # discharge
# discharge_current_on_indices = [5,5,5,5,5,5,5,5,5,5]
# discharge_current_off_indices = [6,6,6,6,6,6,6,6,6,8]
# discharge_cycle_indices = [1,2,3,4,5,6,7,8,9,10] 
# # charge
# charge_current_on_indices = [9,9,9,9,9,9,9,9,9,12]
# charge_current_off_indices = [10,10,10,10,10,10,10,10,10,13]
# charge_cycle_indices = [10,11,12,13,14,15,16,17,18,19]

# Incremental Current OCV - Sample 2 - Data for 45 deg C
# samNum = 2
# T = 45
# df1 = pd.read_excel('../Data/12_09_2015_Incremental OCV test_SP20-3.xlsx',sheet_name='Channel_1-006_1')
# df2 = pd.read_excel('../Data/12_09_2015_Incremental OCV test_SP20-3.xlsx',sheet_name='Channel_1-006_2')
# df3 = pd.read_excel('../Data/12_09_2015_Incremental OCV test_SP20-3.xlsx',sheet_name='Channel_1-006_3')
# df = pd.concat([df1, df2, df3], ignore_index=True)
# # step index and there meanings
# # discharge
# discharge_current_on_indices = [5,5,5,5,5,5,5,5,5,5]
# discharge_current_off_indices = [6,6,6,6,6,6,6,6,6,6]
# discharge_cycle_indices = [1,2,3,4,5,6,7,8,9,10] 
# # discharge previous (used for new Ro calculation)
# discharge_current_on_prev_indices = [5,5,5,5,5,5,5,5,5,5]
# discharge_current_off_prev_indices = [4,6,6,6,6,6,6,6,6,6]
# discharge_cycle_prev_indices = [1,1,2,3,4,5,6,7,8,9] 
# # charge
# charge_current_on_indices = [9,9,9,9,9,9,9,9,12]
# charge_current_off_indices = [10,10,10,10,10,10,10,10,13]
# charge_cycle_indices = [11,12,13,14,15,16,17,18,19]


def double_exponential(t, a1, a2, tau1, tau2):
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)

# Discharge
# print('Discharge analysis')
cycle_ind = 0
for cycle_index in discharge_cycle_indices:
    # print(cycle_ind)
    # print(cycle_index)
    discharge_current_on = discharge_current_on_indices[cycle_ind]
    discharge_current_off = discharge_current_off_indices[cycle_ind]

    # discharge_current_off_prev_step = discharge_current_off_prev_indices[cycle_ind]
    # discharge_current_off_prev_cycle = discharge_cycle_prev_indices[cycle_ind]
    
    data_to_fit = df[(df['Cycle_Index'] == cycle_index) & (df['Step_Index'] == discharge_current_off)]
    data_to_fit_prev = df[(df['Cycle_Index'] == cycle_index) & (df['Step_Index'] == discharge_current_on)]
    # data_to_fit_prev_prev = df[(df['Cycle_Index'] == discharge_current_off_prev_cycle) & (df['Step_Index'] == discharge_current_off_prev_step)]
    y_data_to_fit = data_to_fit['Voltage(V)']
    x_data_to_fit = data_to_fit['Step_Time(s)']
    size = len(y_data_to_fit)

    OCV = data_to_fit['Voltage(V)'].iloc[-1]
    Vo = data_to_fit_prev['Voltage(V)'].iloc[-1]
    Io = data_to_fit_prev['Current(A)'].iloc[-1]
    # Ro = (Vo-y_data_to_fit.iloc[0])/Io
    Ro = (y_data_to_fit.iloc[0]-Vo)/(-Io)
    delta_t = x_data_to_fit.iloc[0]
    absolute_time = data_to_fit['Test_Time(s)'].iloc[0]
    absolute_time_prev = data_to_fit_prev['Test_Time(s)'].iloc[-1]

    # for V_threshold in V_threshold_range:
    x_data_to_fit_prime = x_data_to_fit
    y_data_to_fit_prime = (y_data_to_fit-y_data_to_fit.iloc[size-1])*-1
    y_data_to_fit_prime = y_data_to_fit_prime * -1
    print('Sample')
    print('Io = ', Io)
    print('OCV = ', OCV)
    print('last voltage = ', y_data_to_fit.iloc[size-1])
    print('dV = ', y_data_to_fit.iloc[0]-Vo)
    # print('dV (on other side) = ', data_to_fit_prev_prev['Voltage(V)'].iloc[-1] - data_to_fit_prev['Voltage(V)'].iloc[0])
    print('delta_t (step time) = ', delta_t)
    print('delta_t (absolute time) = ', absolute_time-absolute_time_prev)

    initial_guess = [0.25,0.1,225,2100]

    params, covariance = curve_fit(double_exponential, x_data_to_fit_prime, y_data_to_fit_prime, p0=initial_guess)
    a1, a2, tau1, tau2 = params
    y_fit = double_exponential(x_data_to_fit_prime,*params)

    ss_res = np.sum((y_data_to_fit_prime - y_fit)**2)
    ss_tot = np.sum((y_data_to_fit_prime - np.mean(y_data_to_fit_prime))**2)
    r_squared = 1 - (ss_res / ss_tot)

    solution_found = [OCV, Ro, delta_t, a1, a2, tau1, tau2, r_squared]
    
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
        param_track = solution_found
    else:
        param_track = np.vstack([param_track,solution_found])
    cycle_ind = cycle_ind + 1

# Charge
# print('Charge analysis')
cycle_ind = 0
for cycle_index in charge_cycle_indices:
    # print(cycle_ind)
    # print(cycle_index)
    charge_current_on = charge_current_on_indices[cycle_ind]
    charge_current_off = charge_current_off_indices[cycle_ind]
    data_to_fit = df[(df['Cycle_Index'] == cycle_index) & (df['Step_Index'] == charge_current_off)]
    data_to_fit_prev = df[(df['Cycle_Index'] == cycle_index) & (df['Step_Index'] == charge_current_on)]
    # if cycle_index == charge_cycle_indices[-1]:
    #     print('here')
    #     data_to_fit = df[(df['Cycle_Index'] == cycle_index) & (df['Step_Index'] == last_charge_current_off)]
    #     data_to_fit_prev = df[(df['Cycle_Index'] == cycle_index) & (df['Step_Index'] == last_charge_current_on)]
    # else:
    #     data_to_fit = df[(df['Cycle_Index'] == cycle_index) & (df['Step_Index'] == charge_current_off)]
    #     data_to_fit_prev = df[(df['Cycle_Index'] == cycle_index) & (df['Step_Index'] == charge_current_on)]
    
    y_data_to_fit = data_to_fit['Voltage(V)']
    x_data_to_fit = data_to_fit['Step_Time(s)']
    size = len(y_data_to_fit)

    OCV = data_to_fit['Voltage(V)'].iloc[-1]
    Vo = data_to_fit_prev['Voltage(V)'].iloc[-1]
    Io = data_to_fit_prev['Current(A)'].iloc[-1]
    # Ro = (Vo-y_data_to_fit.iloc[0])/Io
    Ro = (y_data_to_fit.iloc[0]-Vo)/(-Io)
    delta_t = x_data_to_fit.iloc[0]

    # for V_threshold in V_threshold_range:
    x_data_to_fit_prime = x_data_to_fit
    # y_data_to_fit_prime = (y_data_to_fit-y_data_to_fit.iloc[size-1])*-1
    y_data_to_fit_prime = (y_data_to_fit-y_data_to_fit.iloc[size-1])

    initial_guess = [0.25,0.1,225,2100]

    params, covariance = curve_fit(double_exponential, x_data_to_fit_prime, y_data_to_fit_prime, p0=initial_guess)
    a1, a2, tau1, tau2 = params
    y_fit = double_exponential(x_data_to_fit_prime,*params)

    ss_res = np.sum((y_data_to_fit_prime - y_fit)**2)
    ss_tot = np.sum((y_data_to_fit_prime - np.mean(y_data_to_fit_prime))**2)
    r_squared = 1 - (ss_res / ss_tot)

    solution_found = [OCV, Ro, delta_t, a1, a2, tau1, tau2, r_squared]
    
    fig, axs = plt.subplots(2,1,figsize=(8, 5), sharex=True)
    axs[0].plot(x_data_to_fit_prime,y_data_to_fit_prime,label='Data')
    axs[0].plot(x_data_to_fit_prime,y_fit,label='Fit')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Voltage(V)')
    axs[0].set_title(f'Charge Recovery, Cycle Index: {cycle_index}')
    axs[0].legend()
    
    axs[1].plot(x_data_to_fit,y_data_to_fit)
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Voltage(V)')
    axs[1].set_title(f'Original Data, Cycle Index: {cycle_index}')

    # if cycle_ind < 1:
    #     param_track = solution_found
    # else:
    #     param_track = np.vstack([param_track,solution_found])
    
    param_track = np.vstack([param_track,solution_found])
    cycle_ind = cycle_ind + 1

# print(param_track)
df_export = pd.DataFrame(param_track,columns=["OCV","Ro","delta_t","a1","a2","tau1","tau2","r2"])
filename = f"IncrementalOCV_Sample{samNum}_{T}degC_params.csv"
df_export.to_csv(filename, index=False)
plt.show()