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

fig, axs = plt.subplots(figsize=(8, 5), sharex=True)


axs.plot(SOC,OCV,label='OCV vs SOC')
axs.set_title(f'OCV vs SOC')
axs.set_xlabel('SOC')
axs.set_ylabel('OCV (V)')
axs.grid(True)

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
# name = 'DST_25degC_80SOC'
# df = pd.read_excel('../Data/DST/SP2_25C_DST/11_05_2015_SP20-2_DST_80SOC.xls',sheet_name='Channel_1-008')
# data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([7,8]))]
# print(name)

# DST - Data for 45 deg C (50 % SOC)
name = 'DST_45degC_50SOC'
df = pd.read_excel('../Data/DST/SP2_45C_DST/12_11_2015_SP20-2_45C_DST_50SOC.xls',sheet_name='Channel_1-005')
data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([7,8]))]
print(name)

# DST - Data for 45 deg C (80 % SOC)
# name = 'DST_45degC_80SOC'
# df = pd.read_excel('../Data/DST/SP2_45C_DST/12_11_2015_SP20-2_45C_DST_80SOC.xls',sheet_name='Channel_1-005')
# data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([7,8]))]
# print(name)

# series data
t_data = data_to_fit['Test_Time(s)'] - data_to_fit['Test_Time(s)'].iloc[0]
i_data = -data_to_fit['Current(A)']
V_data = data_to_fit['Voltage(V)']
Charge_Capacity = data_to_fit['Charge_Capacity(Ah)']
Discharge_Capacity = data_to_fit['Discharge_Capacity(Ah)']
Qmax = Qnom
Qmax = max(Discharge_Capacity)
Qmin = 0
SOC_charge = (Charge_Capacity-Qmin)/(Qmax-Qmin)
SOC_discharge = (Qmax-Discharge_Capacity)/(Qmax-Qmin)
SOC_data = SOC_discharge
SOC_data = SOC_data.clip(lower=0)
print('SOC from data = ', SOC_data.iloc[0])
print('SOC from voltage = ', SOC_fn(V_data.iloc[0]))
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
    # param_history.append(xk)
    loss_history.append(objective(xk, t_data, i_data, SOC_data, V_data, OCV_fn, dOCV_dSOC_fn, Q))
    param_history.append(np.copy(xk))

def simulate_RC_voltage(t, i_data, R1, C1):
    i_interp = interp1d(t, i_data, fill_value="extrapolate")
    # i_interp = i_data

    def rc_model(V_vec, t):
        i_t = i_interp(t)
        V1 = V_vec
        dV1dt = -V1 / (R1 * C1) + i_t / C1
        return dV1dt

    V0 = 0.0  # initial values for V1 and V2
    V_sol = odeint(rc_model, V0, t)
    V1 = V_sol[:, 0]
    return V1

def compute_model_voltage(t, i_data, SOC_data, OCV_fn, dOCV_dSOC_fn, Q, R0, R1, C1):
    # Interpolate current
    # i_interp = interp1d(t, i_data, fill_value="extrapolate")
    
    # OCV and dOCV/dSOC
    V_ocv = OCV_fn(SOC_data)
    dOCV_dSOC = dOCV_dSOC_fn(SOC_data)
    
    # dV_ocv/dt = dOCV/dSOC * i / Q
    dV_ocv_dt = dOCV_dSOC * i_data / Q

    # V0 = i * R0
    V0 = i_data * R0
    
    # RC network voltages
    V1 = simulate_RC_voltage(t, i_data, R1, C1)

    # Total voltage
    V_model = V_ocv - V0 - V1 
    return V_model

def objective(params, t, i_data, SOC_data, V_measured, OCV_fn, dOCV_dSOC_fn, Q):
    R0, R1, C1 = params
    # R0, R1, C1, R2, C2 = np.exp(params)
    C1 = np.exp(K1*C1)
    V_model = compute_model_voltage(t, i_data, SOC_data, OCV_fn, dOCV_dSOC_fn, Q, R0, R1, C1)

    # loss = (V_model - V_measured)**2
    # loss = np.mean((V_model - V_measured)**2)
    # loss_history.append(loss)
    # return np.mean(loss)
    return np.mean((V_model - V_measured)**2)

def residuals(params, t, i_data, SOC_data, V_measured, OCV_fn, dOCV_dSOC_fn, Q):
    R0, R1, C1 = params
    # R0, R1, C1, R2, C2 = np.exp(params)
    C1 = np.exp(K1*C1)
    V_model = compute_model_voltage(t, i_data, SOC_data, OCV_fn, dOCV_dSOC_fn, Q, R0, R1, C1)

    loss = np.sum((V_measured-V_model)**2)
    loss_history.append(loss)
    param_history.append(params)

    return (V_measured-V_model)

# Wrapper to fit parameters
def fit_battery_model(t, i_data, V_data, SOC_data, OCV_fn, dOCV_dSOC_fn, Q,
                      init_guess,bounds_use):
    # bounds = [(1e-6, 1.0), (1e-6, 1.0), (1e-2, 1e5), (1e-6, 1.0), (1e-2, 1e5)]
    # bounds = [(0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf), (0, np.inf)]
    # bounds = [(np.log(1e-6), np.inf), (np.log(1e-6), np.inf), (np.log(1e-6), np.inf), (np.log(1e-6), np.inf), (np.log(1e-6), np.inf)]
    bounds = bounds_use
    result = minimize(objective, init_guess,
                      args=(t, i_data, SOC_data, V_data, OCV_fn, dOCV_dSOC_fn, Q),
                      bounds=bounds,callback=callback,options={'maxiter': 100})
    return result  # returns R0, R1, C1, R2, C2

def fit_battery_model_leastSquares(t, i_data, V_data, SOC_data, OCV_fn, dOCV_dSOC_fn, Q,
                      init_guess,bounds_use):
    bounds = bounds_use
    result = least_squares(residuals, init_guess,
                    args=(t, i_data, SOC_data, V_data, OCV_fn, dOCV_dSOC_fn, Q),
                    bounds=bounds)

    return result  # returns R0, R1, C1, R2, C2

# Example usage:
# Define OCV(SOC) and dOCV_dSOC(SOC) functions from your data
# from scipy.interpolate import interp1d
# OCV_fn = interp1d(SOC_points, OCV_points, fill_value="extrapolate")
# dOCV_dSOC_fn = interp1d(SOC_points, dOCV_dSOC_points, fill_value="extrapolate")

# Fit:
Q = Qnom
K1 = 100
init_guess = [0.01, 0.01, 100.0]
init_guess = np.log(init_guess)
init_guess = [0.01, 0.01, np.log(2000.0)/K1]
bounds_use = [(1e-6, np.inf), (1e-6, np.inf), (np.log(1e-6), np.inf)]
# result = fit_battery_model(t_data, i_data, V_data, SOC_data, OCV_fn, dOCV_dSOC_fn, Q, init_guess, bounds_use)

bounds_use_leastSquares = ([1e-6, 1e-6, np.log(1e-6)],[np.inf, np.inf, np.inf])
result = fit_battery_model_leastSquares(t_data, i_data, V_data, SOC_data, OCV_fn, dOCV_dSOC_fn, Q, init_guess, bounds_use_leastSquares)

R0, R1, C1 = result.x

# R0, R1, C1, R2, C2 = np.exp(result.x)
C1 = np.exp(K1*C1)
print('R_0 = ', R0)
print('R_1 = ', R1)
print('C_1 = ', C1)

param_track = [R0, R1, C1]
# df_export = pd.DataFrame([param_track],columns=["Ro","R1","C1","R2","C2"])
# filename = f"{name}.csv"
# df_export.to_csv(filename, index=False)

# Plot:
V_fit = compute_model_voltage(t_data, i_data, SOC_data, OCV_fn, dOCV_dSOC_fn, Q, R0, R1, C1)

RMS_error = (1/len(V_fit)*np.sum((V_data-V_fit)**2))**(1/2)
print('RMS Error = ', RMS_error)

fig, axs = plt.subplots(figsize=(8, 5), sharex=True)
axs.plot(t_data, V_data, label='Measured')
axs.plot(t_data, V_fit, label='Model')
axs.set_xlabel('Time (s)')
axs.set_ylabel('Voltage (V)')
axs.set_title(f'Data vs Model')
axs.legend()

fig, axs = plt.subplots(figsize=(8, 5), sharex=True)
axs.plot(loss_history)
axs.set_xlabel("Iteration")
axs.set_ylabel("Loss (MSE)")
axs.set_title("Optimization Loss History")
axs.grid(True)

fig, axs = plt.subplots(3,1,figsize=(8, 5), sharex=True)
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


plt.show()