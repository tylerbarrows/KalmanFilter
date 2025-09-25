##
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# Assume these are provided as NumPy arrays:
# t_data, i_data, V_measured, SOC_data
# OCV vs SOC and dOCV/dSOC vs SOC functions
# Q = battery capacity in Ah, convert to As (Coulombs): Q = Q_Ah * 3600
Qnom = 2000 # mAh
Qnom = Qnom/1000 # Ah

# Loading SOV vs OCV
# SOC_OCV = np.load('OCV_dOCV_dSOC_SOC_LowCurrent.npz')
SOC_OCV = np.load('OCV_dOCV_dSOC_SOC_Incremental.npz')

OCV = SOC_OCV['v1']
dOCV_dSOC = SOC_OCV['v2']
SOC = SOC_OCV['v3']

# 0 deg C
# SOC_OCV_Incremental_sam1_0degC = np.load('OCV_dOCV_dSOC_SOC_Incremental_sam1_0degC.npz')
# SOC_OCV_Incremental_sam2_0degC = np.load('OCV_dOCV_dSOC_SOC_Incremental_sam2_0degC.npz')
# OCV_1 = SOC_OCV_Incremental_sam1_0degC['v1']
# dOCV_1 =SOC_OCV_Incremental_sam1_0degC['v1']
# SOC_1 = SOC_OCV_Incremental_sam1_0degC['v3']
# OCV_2 = SOC_OCV_Incremental_sam2_0degC['v1']
# dOCV_2 = SOC_OCV_Incremental_sam2_0degC['v2']
# SOC_2 = SOC_OCV_Incremental_sam2_0degC['v3']
# 25 deg C
# SOC_OCV_Incremental_sam1_25degC = np.load('OCV_dOCV_dSOC_SOC_Incremental_sam1_25degC.npz')
# SOC_OCV_Incremental_sam2_25degC = np.load('OCV_dOCV_dSOC_SOC_Incremental_sam2_25degC.npz')
# OCV_1 = SOC_OCV_Incremental_sam1_25degC['v1']
# dOCV_1 =SOC_OCV_Incremental_sam1_25degC['v1']
# SOC_1 = SOC_OCV_Incremental_sam1_25degC['v3']
# OCV_2 = SOC_OCV_Incremental_sam2_25degC['v1']
# dOCV_2 = SOC_OCV_Incremental_sam2_25degC['v2']
# SOC_2 = SOC_OCV_Incremental_sam2_25degC['v3']
# 45 deg C
# SOC_OCV_Incremental_sam1_45degC = np.load('OCV_dOCV_dSOC_SOC_Incremental_sam1_45degC.npz')
# SOC_OCV_Incremental_sam2_45degC = np.load('OCV_dOCV_dSOC_SOC_Incremental_sam2_45degC.npz')
# OCV_1 = SOC_OCV_Incremental_sam1_45degC['v1']
# dOCV_1 =SOC_OCV_Incremental_sam1_45degC['v1']
# SOC_1 = SOC_OCV_Incremental_sam1_45degC['v3']
# OCV_2 = SOC_OCV_Incremental_sam2_45degC['v1']
# dOCV_2 = SOC_OCV_Incremental_sam2_45degC['v2']
# SOC_2 = SOC_OCV_Incremental_sam2_45degC['v3']

# OCV = 1/2*(OCV_1+OCV_2)
# dOCV_dSOC = 1/2*(dOCV_1+dOCV_2)
# SOC = SOC_1

SOC_fn = interp1d(OCV, SOC)  # finding SOC based on OCV
OCV_fn = interp1d(SOC, OCV)
dOCV_dSOC_fn = interp1d(SOC,dOCV_dSOC)

# Loading data

# DST - Data for 0 deg C (50 % SOC)
# name = 'DST_0degC_50SOC'
# df = pd.read_excel('../Data/DST/SP2_0C_DST/02_24_2016_SP20-2_0C_DST_50SOC.xls',sheet_name='Channel_1-006')
# data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([7,8]))]
# print(name)

# DST - Data for 0 deg C (80 % SOC)
# name = 'DST_0degC_80SOC'
# df = pd.read_excel('../Data/DST/SP2_0C_DST/02_24_2016_SP20-2_0C_DST_80SOC.xls',sheet_name='Channel_1-006')
# data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([7,8]))]
# print(name)

# DST - Data for 25 deg C (50 % SOC)
# name = 'DST_25degC_50SOC'
# df = pd.read_excel('../Data/DST/SP2_25C_DST/11_05_2015_SP20-2_DST_50SOC.xls',sheet_name='Channel_1-008')
# data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([7,8]))]
# print(name)

# DST - Data for 25 deg C (80 % SOC)
name = 'DST_25degC_80SOC'
df = pd.read_excel('../Data/DST/SP2_25C_DST/11_05_2015_SP20-2_DST_80SOC.xls',sheet_name='Channel_1-008')
data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([7,8]))]
print(name)

# DST - Data for 45 deg C (50 % SOC)
# name = 'DST_45degC_50SOC'
# df = pd.read_excel('../Data/DST/SP2_45C_DST/12_11_2015_SP20-2_45C_DST_50SOC.xls',sheet_name='Channel_1-005')
# data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([7,8]))]
# print(name)

# DST - Data for 45 deg C (80 % SOC)
# name = 'DST_45degC_80SOC'
# df = pd.read_excel('../Data/DST/SP2_45C_DST/12_11_2015_SP20-2_45C_DST_80SOC.xls',sheet_name='Channel_1-005')
# data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([7,8]))]
# print(name)

# series data
t_data = data_to_fit['Test_Time(s)'].to_numpy()
t_data = t_data - t_data[0]
i_data = -data_to_fit['Current(A)'].to_numpy()
V_data = data_to_fit['Voltage(V)'].to_numpy()
Charge_Capacity = data_to_fit['Charge_Capacity(Ah)'].to_numpy()
Discharge_Capacity = data_to_fit['Discharge_Capacity(Ah)'].to_numpy()
Qmax = Qnom
# Qmax = max(Discharge_Capacity)
Qmin = 0
SOC_charge = (Charge_Capacity-Qmin)/(Qmax-Qmin)
SOC_discharge = (Qmax-Discharge_Capacity)/(Qmax-Qmin)
SOC_data = SOC_discharge
SOC_data[SOC_data<0] = 0
print('SOC from data = ', SOC_data[0])
print('SOC from voltage = ', SOC_fn(V_data[0]))
print('min. SOC = ', min(SOC_data))

# Useful figure for looking at SOC
fig, axs = plt.subplots(4, 1, figsize=(8, 5), sharex=True)
size_for_markers_plot = 2
size_for_markers_scatter = 22

axs[0].plot(t_data, V_data, marker='o', markersize=size_for_markers_plot)
axs[0].set_title(f'Cropped data')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('Voltage (V)')
axs[0].grid(True)

axs[1].plot(t_data, i_data, marker='o', markersize=size_for_markers_plot)
# axs[1].set_title(f'Cropped data')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Current (A)')
axs[1].grid(True)

axs[2].plot(t_data, Charge_Capacity, marker='o', markersize=size_for_markers_plot,label='Charge Capacity')
axs[2].plot(t_data, Discharge_Capacity, marker='o', markersize=size_for_markers_plot,label='Discharge Capacity')
# axs[2].set_title(f'Cropped data')
axs[2].set_xlabel('Time')
axs[2].set_ylabel('Charge Capacity (Ah)')
axs[2].legend()
axs[2].grid(True)

axs[3].plot(t_data, SOC_charge, marker='o', markersize=size_for_markers_plot,label='Charge Capacity')
axs[3].plot(t_data, SOC_discharge, marker='o', markersize=size_for_markers_plot,label='Discharge Capacity')
# axs[3].set_title(f'Cropped data')
axs[3].set_xlabel('Time')
axs[3].set_ylabel('SOC')
axs[3].legend()
axs[3].grid(True)

loss_history = []
param_history = []

def callback(xk):
    loss_history.append(objective(xk,t_data, i_data, SOC_data, V_data, OCV_fn, dOCV_dSOC_fn, Q))
    param_history.append(np.copy(xk))

def simulate_RC_voltage(t, i_data, R1, C1, R2, C2):
    i_interp = interp1d(t, i_data, fill_value="extrapolate")

    def rc_model(V_vec, t): 
        i_t = i_interp(t)
        V1, V2 = V_vec
        dV1dt = -V1 / (R1 * C1) + i_t / C1
        dV2dt = -V2 / (R2 * C2) + i_t / C2
        return [dV1dt, dV2dt]
    
    V0 = [0.0, 0.0]  # initial values for V1 and V2
    V_sol = odeint(rc_model, V0, t)
    V1 = V_sol[:, 0]
    V2 = V_sol[:, 1]

    # def rc_model_1(V_vec, t): 
    #     i_t = i_interp(t)
    #     V1 = V_vec
    #     dV1dt = -V1 / (R1 * C1) + i_t / C1
    #     return dV1dt

    # def rc_model_2(V_vec, t): 
    #     i_t = i_interp(t)
    #     V2 = V_vec
    #     dV2dt = -V2 / (R2 * C2) + i_t / C2
    #     return dV2dt

    # V_rc_0 = 0.0 # initial values for V1 and V2
    # V1 = odeint(rc_model_1, V_rc_0, t)
    # V1 = V1[:,0]
    # V2 = odeint(rc_model_2, V_rc_0, t)
    # V2 = V2[:,0]

    return V1, V2

def simulate_RC_voltage_online(t,i_data,R1,C1,R2,C2):
    V1 = np.zeros_like(i_data, dtype=float)
    V2 = np.zeros_like(i_data, dtype=float)

    V1[0] = 0
    V2[0] = 0

    for k in range(1,len(t)):
        dt = t[k] - t[k-1]
        V1[k] = V1[k-1]*np.exp(-dt/R1/C1) + i_data[k-1]*R1*(1-np.exp(-dt/R1/C1))
        V2[k] = V2[k-1]*np.exp(-dt/R2/C2) + i_data[k-1]*R2*(1-np.exp(-dt/R2/C2))

    return V1, V2

def compute_model_voltage(t, i_data, SOC_data, OCV_fn, dOCV_dSOC_fn, Q, R0, R1, C1, R2, C2):
    # Interpolate current
    # i_interp = interp1d(t, i_data, fill_value="extrapolate")
    
    # OCV 
    V_ocv = OCV_fn(SOC_data)

    # V0 = i * R0
    V0 = i_data * R0
    
    # RC network voltages
    V1, V2 = simulate_RC_voltage(t, i_data, R1, C1, R2, C2)
    # V1, V2 = simulate_RC_voltage_online(t, i_data, R1, C1, R2, C2)

    # Total voltage
    V_model = V_ocv - V0 - V1 - V2
    return V_model, V_ocv, V0, V1, V2

def objective(params, t, i_data, SOC_data, V_measured, OCV_fn, dOCV_dSOC_fn, Q):
    R0, R1, C1, R2, C2 = params
    # R0, R1, C1, R2, C2 = np.exp(params)
    C1 = np.exp(K1*C1)
    C2 = np.exp(K2*C2)
    V_model, V_ocv, V0, V1, V2 = compute_model_voltage(t, i_data, SOC_data, OCV_fn, dOCV_dSOC_fn, Q, R0, R1, C1, R2, C2)

    return np.mean((V_model - V_measured)**2)

def residuals(params, t, i_data, SOC_data, V_measured, OCV_fn, dOCV_dSOC_fn, Q):
    R0, R1, C1, R2, C2 = params
    # R0, R1, C1, R2, C2 = np.exp(params)
    C1 = np.exp(K1*C1)
    C2 = np.exp(K2*C2)
    # V_model, V_ocv, V0, V1, V2 = compute_model_voltage(t, i_data, SOC_data, OCV_fn, dOCV_dSOC_fn, Q, R0, R1, C1, R2, C2)
    V_model, V_ocv, V0, V1, V2 = online_voltage_fit(t, i_data, SOC_data, OCV_fn, Q, R0, R1, C1, R2, C2)

    loss = np.sum((V_measured-V_model)**2)
    loss_history.append(loss)
    param_history.append(params)

    return (V_measured-V_model)

# Wrapper to fit parameters
def fit_battery_model_minimize(t, i_data, V_data, SOC_data, OCV_fn, dOCV_dSOC_fn, Q,
                      init_guess,bounds_use):
    bounds = bounds_use
    result = minimize(objective, init_guess,
                      args=(t, i_data, SOC_data, V_data, OCV_fn, dOCV_dSOC_fn, Q),
                      bounds=bounds,callback=callback)
    return result  # returns R0, R1, C1, R2, C2

def fit_battery_model_leastSquares(t, i_data, V_data, SOC_data, OCV_fn, dOCV_dSOC_fn, Q,
                      init_guess,bounds_use):
    bounds = bounds_use
    result = least_squares(residuals, init_guess,
                    args=(t, i_data, SOC_data, V_data, OCV_fn, dOCV_dSOC_fn, Q),
                    bounds=bounds)

    return result  # returns R0, R1, C1, R2, C2

def online_voltage_fit(t, i_data, SOC_data, OCV_fn, Q, R0, R1, C1, R2, C2):

    V = np.zeros_like(i_data, dtype=float)

    V_OCV = np.zeros_like(i_data, dtype=float)
    V_OCV[0] = OCV_fn(SOC_data[0])

    V0 = np.zeros_like(i_data, dtype=float)
    V0[0] = i_data[0] * R0

    V1 = np.zeros_like(i_data, dtype=float)
    V1[0] = 0

    V2 = np.zeros_like(i_data, dtype=float)
    V2[0] = 0
    
    V[0] = V_OCV[0] - V0[0] - V1[0] - V2[0]
    # V[0] = V2

    for k in range(1,len(t)):
        dt = t[k] - t[k-1]
        V_OCV[k] = OCV_fn(SOC_data[k])
        V0[k] = i_data[k] * R0
        
        V1[k] = V1[k-1]*np.exp(-dt/R1/C1) + i_data[k-1]*R1*(1-np.exp(-dt/R1/C1))
        V2[k] = V2[k-1]*np.exp(-dt/R2/C2) + i_data[k-1]*R2*(1-np.exp(-dt/R2/C2))

        # V1[k] = V1[k-1] - V1[k-1]*dt/R1/C1 + i_data[k-1]*dt/C1
        # V2[k] = V2[k-1] - V2[k-1]*dt/R2/C2 + i_data[k-1]*dt/C2

        V[k] = V_OCV[k] - V0[k] - V1[k] - V2[k]

        # V[k] = V_OCV
        # V[k] = V2

    return V, V_OCV, V0, V1, V2


# Example usage:
# Define OCV(SOC) and dOCV_dSOC(SOC) functions from your data
# from scipy.interpolate import interp1d
# OCV_fn = interp1d(SOC_points, OCV_points, fill_value="extrapolate")
# dOCV_dSOC_fn = interp1d(SOC_points, dOCV_dSOC_points, fill_value="extrapolate")

# Fit:
Q = Qnom
K1 = 10
K2 = 10

# ln space for C1 and C2
init_guess = [0.01, 0.01, np.log(100.0)/K1, 0.01, np.log(1000.0)/K2]
init_guess = [0.01, 0.01, np.log(100.0)/K1, 0.01, np.log(1000.0)/K2]
# init_guess = [0.1, 0.01, 100.0, 0.05, 1000]

bounds_use_minimize = [(1e-6, np.inf), (1e-6, np.inf), (np.log(1e-6), np.inf), (1e-6, np.inf), (np.log(1e-6), np.inf)]
bounds_use_minimize = [(1e-4, np.inf), (1e-4, np.inf), (np.log(1e-4), np.inf), (1e-4, np.inf), (np.log(1e-4), np.inf)]
# result = fit_battery_model_minimize(t_data, i_data, V_data, SOC_data, OCV_fn, dOCV_dSOC_fn, Q, init_guess, bounds_use_minimize)

bounds_use_leastSquares = ([1e-6, 1e-6, np.log(1e-6), 1e-6, np.log(1e-6)],[np.inf, np.inf, np.inf, np.inf, np.inf])
result = fit_battery_model_leastSquares(t_data, i_data, V_data, SOC_data, OCV_fn, dOCV_dSOC_fn, Q, init_guess, bounds_use_leastSquares)

R0, R1, C1, R2, C2 = result.x
# R0, R1, C1, R2, C2 = np.exp(result.x)
C1 = np.exp(K1*C1)
C2 = np.exp(K2*C2)
print('R_0 = ', R0)
print('R_1 = ', R1)
print('C_1 = ', C1)
print('R_2 = ', R2)
print('C_2 = ', C2)

V_fit, V_ocv_fit, V0_fit, V1_fit, V2_fit = compute_model_voltage(t_data, i_data, SOC_data, OCV_fn, dOCV_dSOC_fn, Q, R0, R1, C1, R2, C2)
residuals = (V_data - V_fit)
SS_res = np.sum(residuals**2)
SS_tot = np.sum((V_data-np.mean(V_data))**2)
R2_fit = 1-SS_res/SS_tot # this is where the error is coming from
RMS = SS_res * 1/np.sqrt(len(V_data))

V_fit_online, V_ocv_fit_online, V0_fit_online, V1_fit_online, V2_fit_online = online_voltage_fit(t_data, i_data, SOC_data, OCV_fn, Q, R0,R1,C1,R2,C2)

# ####################################################
# # troubleshooting
# V1_check, V2_check = simulate_RC_voltage_online(t_data,i_data,R1,C1,R2,C2)

# V1 = np.zeros_like(V_data, dtype=float)
# V2 = np.zeros_like(V_data, dtype=float)

# V1[0] = 0
# V2[0] = 0

# for k in range(1,len(t_data)):
#     dt = t_data[k] - t_data[k-1]
#     V1[k] = V1[k-1]*np.exp(-dt/R1/C1) + i_data[k-1]*R1*(1-np.exp(-dt/R1/C1))
#     V2[k] = V2[k-1]*np.exp(-dt/R2/C2) + i_data[k-1]*R2*(1-np.exp(-dt/R2/C2))

# fig, axs = plt.subplots(2,1,figsize=(8, 5), sharex=True)

# # axs[0].plot(t_data, V1_check, marker='o', markersize=size_for_markers_plot,label='RC Model')
# # axs[0].plot(t_data, V1, marker='x', markersize=size_for_markers_plot,label='Manual')
# axs[0].plot(t_data, V1_check-V1, marker='o', markersize=size_for_markers_plot,label='Error')
# axs[0].set_xlabel("Time (s)")
# axs[0].set_ylabel("V1")
# axs[0].set_title("V1 check")
# axs[0].grid(True)
# axs[0].legend()

# # axs[1].plot(t_data, V2_check, marker='o', markersize=size_for_markers_plot,label='RC Model')
# # axs[1].plot(t_data, V2, marker='x', markersize=size_for_markers_plot,label='Manual')
# axs[1].plot(t_data, V2_check-V2, marker='o', markersize=size_for_markers_plot,label='Error')
# axs[1].set_xlabel("Time (s)")
# axs[1].set_ylabel("V2")
# axs[1].set_title("V2 check")
# axs[1].grid(True)
# axs[1].legend()
# ####################################################

# ####################################################
# # troubleshooting
# V_OCV_1 = OCV_fn(SOC_data)
# V_OCV_2 = np.zeros_like(V_data, dtype=float)
# V_OCV_2[0] = OCV_fn(SOC_data[0])
# for k in range(1,len(V_data)):
#     V_OCV_2[k] = OCV_fn(SOC_data[k])

# fig, axs = plt.subplots(figsize=(8, 5), sharex=True)

# # axs.plot(t_data, V_OCV_1, marker='o', markersize=size_for_markers_plot,label='Interpolate Vector')
# # axs.plot(t_data, V_OCV_2, marker='x', markersize=size_for_markers_plot,label='Interpolate Individual')
# axs.plot(t_data, V_OCV_1-V_OCV_2, marker='x', markersize=size_for_markers_plot,label='Error')
# axs.set_xlabel("Time (s)")
# axs.set_ylabel("OCV")
# axs.set_title("OCV check")
# axs.grid(True)
# axs.legend()
# ####################################################

# ####################################################
# # troubleshooting
# V0_1 = i_data * R0
# V0_2 = np.zeros_like(V_data, dtype=float)
# V0_2[0] = i_data[0] * R0
# for k in range(1,len(V_data)):
#     V0_2[k] = i_data[k] * R0

# # axs.plot(t_data, V0_1, marker='o', markersize=size_for_markers_plot,label='Interpolate Vector')
# # axs.plot(t_data, V0_2, marker='x', markersize=size_for_markers_plot,label='Interpolate Individual')
# axs.plot(t_data, V0_1-V0_2, marker='x', markersize=size_for_markers_plot,label='Error')
# axs.set_xlabel("Time (s)")
# axs.set_ylabel("V0")
# axs.set_title("V0 check")
# axs.grid(True)
# axs.legend()
# ####################################################

param_and_error_track = [R0, R1, C1, R2, C2, R2_fit, RMS]
df_export = pd.DataFrame([param_and_error_track],columns=["Ro","R1","C1","R2","C2","R2 fit","RMS"])
# filename = f"{name}.csv"
# df_export.to_csv(filename, index=False)

print('RMS = ', RMS)
print('R2_fit = ', R2_fit)

# Plot:

fig, axs = plt.subplots(figsize=(8, 5), sharex=True)
axs.plot(t_data, V_data, label='Measured')
axs.plot(t_data, V_fit, label='Model')
# axs.plot(t_data, V_fit_online, label='Online Model')
axs.set_xlabel('Time (s)')
axs.set_ylabel('Voltage (V)')
axs.set_title(f'Data vs Model')
axs.legend()

fig, axs = plt.subplots(figsize=(8, 5), sharex=True)
# axs.plot(t_data, V_data, label='Measured')
# axs.plot(t_data, V_fit, label='Model')
axs.plot(t_data, V_fit_online, label='Online Model')
axs.set_xlabel('Time (s)')
axs.set_ylabel('Voltage (V)')
axs.set_title(f'Data vs Model')
axs.legend()

fig, axs = plt.subplots(5,1,figsize=(8, 5), sharex=True)
axs[0].plot(t_data, V_fit, label='Model')
axs[0].plot(t_data, V_fit_online, label='Online Model')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Voltage (V)')
axs[0].set_title(f'V (fit vs online fit)')
axs[0].legend()

axs[1].plot(t_data, V_ocv_fit, label='Model')
axs[1].plot(t_data, V_ocv_fit_online, label='Online Model')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Voltage (V)')
axs[1].set_title(f'OCV (fit vs online fit)')
axs[1].legend()

axs[2].plot(t_data, V0_fit, label='Model')
axs[2].plot(t_data, V0_fit_online, label='Online Model')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('Voltage (V)')
axs[2].set_title(f'V0 (fit vs online fit)')
axs[2].legend()

axs[3].plot(t_data, V1_fit, label='Model')
axs[3].plot(t_data, V1_fit_online, label='Online Model')
axs[3].set_xlabel('Time (s)')
axs[3].set_ylabel('Voltage (V)')
axs[3].set_title(f'V1 (fit vs online fit)')
axs[3].legend()

axs[4].plot(t_data, V2_fit, label='Model')
axs[4].plot(t_data, V2_fit_online, label='Online Model')
axs[4].set_xlabel('Time (s)')
axs[4].set_ylabel('Voltage (V)')
axs[4].set_title(f'V2 (fit vs online fit)')
axs[4].legend()

fig, axs = plt.subplots(figsize=(8, 5), sharex=True)
axs.plot(loss_history)
axs.set_xlabel("Iteration")
axs.set_ylabel("Loss (MSE)")
axs.set_title("Optimization Loss History")
axs.grid(True)

fig, axs = plt.subplots(5,1,figsize=(8, 5), sharex=True)
# axs[0].plot([np.exp(p[0]) for p in param_history],label='R0')
axs[0].plot([p[0] for p in param_history],label='R0')
axs[0].set_xlabel("Iteration")
axs[0].set_ylabel("R0")
axs[0].set_title("Parameter History")
axs[0].grid(True)

# axs[1].plot([np.exp(p[1]) for p in param_history],label='R1')
axs[1].plot([p[1] for p in param_history],label='R1')
axs[1].set_xlabel("Iteration")
axs[1].set_ylabel("R1")
# axs[1].set_title("Parameter History")
axs[1].grid(True)

axs[2].plot([np.exp(K1*p[2]) for p in param_history],label='C1')
# axs[2].plot([p[2] for p in param_history],label='C1')
axs[2].set_xlabel("Iteration")
axs[2].set_ylabel("C1")
# axs[2].set_title("Parameter History")
axs[2].grid(True)

# axs[3].plot([np.exp(p[3]) for p in param_history],label='R2')
axs[3].plot([p[3] for p in param_history],label='R2')
axs[3].set_xlabel("Iteration")
axs[3].set_ylabel("R2")
# axs[3].set_title("Parameter History")
axs[3].grid(True)

axs[4].plot([np.exp(K2*p[4]) for p in param_history],label='C2')
# axs[4].plot([p[4] for p in param_history],label='C2')
axs[4].set_xlabel("Iteration")
axs[4].set_ylabel("C2")
# axs[4].set_title("Parameter History")
axs[4].grid(True)


plt.show()