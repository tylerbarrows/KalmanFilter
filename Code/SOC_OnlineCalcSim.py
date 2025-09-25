## Abstract

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

fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
size_for_markers_plot = 2
size_for_markers_scatter = 22

axs[0].plot(SOC, OCV, marker='o', markersize=size_for_markers_plot)
axs[0].set_title(f'OCV vs SOC')
axs[0].set_xlabel('SOC')
axs[0].set_ylabel('OCV (V)')
axs[0].grid(True)

axs[1].plot(SOC, dOCV_dSOC, marker='o', markersize=size_for_markers_plot)
axs[1].set_title(f'dOCV/dSOC vs SOC')
axs[1].set_xlabel('SOC')
axs[1].set_ylabel('dOCV/dSOC')
axs[1].grid(True)

SOC_fn = interp1d(OCV, SOC)  # finding SOC based on OCV
OCV_fn = interp1d(SOC, OCV)
dOCV_dSOC_fn = interp1d(SOC,dOCV_dSOC)

# DST - Data for 0 deg C (50 % SOC)
# name = 'DST_0degC_50SOC'
# T = 0
# df = pd.read_excel('../Data/DST/SP2_0C_DST/02_24_2016_SP20-2_0C_DST_50SOC.xls',sheet_name='Channel_1-006')
# data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([7,8]))]
# print(name)

# FUDS - Data for 0 deg C (50 % SOC)
# name = 'FUDS_0degC_50SOC'
# T = 0
# df = pd.read_excel('../Data/FUDS/SP2_0C_FUDS/02_25_2016_SP20-2_0C_FUDS_50SOC.xls',sheet_name='Channel_1-006')
# data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([7,8]))]
# # data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([6,7,8]))]
# print(name)

# FUDS - Data for 0 deg C (80 % SOC)
# name = 'FUDS_0degC_80SOC'
# T = 0
# df = pd.read_excel('../Data/FUDS/SP2_0C_FUDS/02_25_2016_SP20-2_0C_FUDS_80SOC.xls',sheet_name='Channel_1-006')
# data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([7,8]))]
# # data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([6,7,8]))]
# print(name)

# FUDS - Data for 25 deg C (50 % SOC)
# name = 'FUDS_25degC_50SOC'
# T = 25
# df = pd.read_excel('../Data/FUDS/SP2_25C_FUDS/11_09_2015_SP20-2_FUDS_50SOC.xls',sheet_name='Channel_1-008')
# data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([7,8]))]
# # data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([6,7,8]))]
# print(name)

# FUDS - Data for 25 deg C (80 % SOC)
name = 'FUDS_25degC_80SOC'
T = 0
df = pd.read_excel('../Data/FUDS/SP2_25C_FUDS/11_06_2015_SP20-2_FUDS_80SOC.xls',sheet_name='Channel_1-008')
data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([7,8]))]
# data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([6,7,8]))]
print(name)

# FUDS - Data for 45 deg C (50 % SOC)
# name = 'FUDS_45degC_50SOC'
# T = 45
# df = pd.read_excel('../Data/FUDS/SP2_45C_FUDS/12_15_2015_SP20-2_45C_FUDS_50SOC.xls',sheet_name='Channel_1-005')
# data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([7,8]))]
# # data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([6,7,8]))]
# print(name)

# FUDS - Data for 45 deg C (80 % SOC)
# name = 'FUDS_45degC_80SOC'
# T = 45
# df = pd.read_excel('../Data/FUDS/SP2_45C_FUDS/12_15_2015_SP20-2_45C_FUDS_80SOC.xls',sheet_name='Channel_1-005')
# data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([7,8]))]
# # data_to_fit = df[(df['Cycle_Index'] == 1) & (df['Step_Index'].isin([6,7,8]))]
# print(name)

# series data
t_data = data_to_fit['Test_Time(s)'].to_numpy()
t_data = t_data - t_data[0]
i_data = -data_to_fit['Current(A)'].to_numpy()
V_data = data_to_fit['Voltage(V)'].to_numpy()
Charge_Capacity = data_to_fit['Charge_Capacity(Ah)'].to_numpy()
Discharge_Capacity = data_to_fit['Discharge_Capacity(Ah)'].to_numpy()
Qmax = Qnom
Qmax = max(Discharge_Capacity)
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

# Initial SOC
SOC_o = SOC_data[0]
SOC_o = SOC_fn(V_data[0])

# Coulomb counting
def coulomb_counting(time,current,SOC_o,Qnom):
    SOC = np.zeros_like(current, dtype=float)
    SOC[0] = SOC_o

    for k in range(1,len(time)):
        dt = time[k] - time[k-1]
        SOC[k] = SOC[k-1] - current[k-1] * dt / Qnom / 3600

    return SOC

def OCV_SOC_lookup(voltage,OCV,SOC):

    SOC_fn_here = interp1d(OCV, SOC, fill_value=(0,1), bounds_error=False)  

    SOC = SOC_fn_here(voltage)

    return SOC

def kalman_fitler(time,current,voltage,OCV_array,dOCV_dSOC_array,SOC_array,Qnom,SOC_o,P1,sigma_bar,params,method_flag):
    R0 = params[0]
    R1 = params[1]
    C1 = params[2]
    R2 = params[3]
    C2 = params[4]

    dOCV_dSOC_fn_here = interp1d(SOC_array, dOCV_dSOC_array, fill_value=(dOCV_dSOC_array[0],dOCV_dSOC_array[-1]), bounds_error=False)
    OCV_fn_here = interp1d(SOC_array, OCV_array, fill_value=(OCV_array[0],OCV_array[-1]), bounds_error=False)  

    P = P1
    sigma_SOC = sigma_bar[0]
    sigma_V = sigma_bar[1]
    sigma_i = sigma_bar[2]
    sigma_Q = sigma_bar[3]
    sigma_R0 = sigma_bar[4]
    sigma_R1 = sigma_bar[5]
    sigma_C1 = sigma_bar[6]
    sigma_R2 = sigma_bar[7]
    sigma_C1 = sigma_bar[8]

    SOC = np.zeros_like(current, dtype=float)
    SOC[0] = SOC_o
    y = np.zeros_like(current, dtype=float)
    y[0] = V_data[0]
    yocv = np.zeros_like(current, dtype=float)
    yocv[0] = V_data[0]
    y0 = np.zeros_like(current, dtype=float)
    y0[0] = 0
    y1 = np.zeros_like(current, dtype=float)
    y1[0] = 0
    y2 = np.zeros_like(current, dtype=float)
    y2[0] = 0
    x = np.array([[SOC[0]],[0],[0]])
    K_track = []
    K_initial = np.array([[0],[0],[0]])
    K_track.append(K_initial)

    for k in range(1,len(time)):
        dt = time[k] - time[k-1]

        if method_flag == 0:
            F = np.array([[1,0,0],[0,1-dt/R1/C1,0],[0,0,1-dt/R2/C2]])
            B = np.array([[-dt/Qnom/3600], [dt/C1], [dt/C2]])

            Q1 = np.power(-dt/Qnom/3600*sigma_i,2)+np.power(dt/np.power(Qnom,2)/3600*current[k-1]*sigma_Q,2)
            # including sigma_V
            # Q2 = np.power((1-dt/R1/C1)*sigma_V,2)+np.power(dt/R1/R1/C1*x[1][0]*sigma_R1,2)+np.power(dt/R1/C1/C1*x[1][0]*sigma_C1-dt/C1/C1*current[k-1]*sigma_C1,2)+np.power(dt/C1*sigma_i,2)
            # Q3 = np.power((1-dt/R2/C2)*sigma_V,2)+np.power(dt/R2/R2/C2*x[2][0]*sigma_R2,2)+np.power(dt/R2/C2/C2*x[2][0]*sigma_C2-dt/C2/C2*current[k-1]*sigma_C2,2)+np.power(dt/C2*sigma_i,2)
            # not including sigma_V
            Q2 = np.power(dt/R1/R1/C1*x[1][0]*sigma_R1,2)+np.power(dt/R1/C1/C1*x[1][0]*sigma_C1-dt/C1/C1*current[k-1]*sigma_C1,2)+np.power(dt/C1*sigma_i,2)
            Q3 = np.power(dt/R2/R2/C2*x[2][0]*sigma_R2,2)+np.power(dt/R2/C2/C2*x[2][0]*sigma_C2-dt/C2/C2*current[k-1]*sigma_C2,2)+np.power(dt/C2*sigma_i,2)

            Q = np.array([[Q1,0,0],[0,Q2,0],[0,0,Q3]])

            R = np.array([np.power(sigma_V,2)])
        else:
            F = np.array([[1,0,0],[0,np.exp(-dt/R1/C1),0],[0,0,np.exp(-dt/R2/C2)]])
            B = np.array([[-dt/Qnom/3600], [R1*(1-np.exp(-dt/R1/C1))], [R2*(1-np.exp(-dt/R2/C2))]])

            Q1 = np.power(dt/Qnom/3600*sigma_i,2)+np.power(dt/np.power(Qnom,2)/3600*current[k-1]*sigma_Q,2)
            # including sigma_V
            # Q2 = np.power(np.exp(-dt/R1/C1)*sigma_V,2)+np.power(dt/R1/R1/C1*np.exp(-dt/R1/C1)*x[1][0]*sigma_R1+1*(1-np.exp(-dt/R1/C1))*current[k-1]*sigma_R1-R1*dt/R1/R1/C1*np.exp(-dt/R1/C1)*current[k-1]*sigma_R1,2)+np.power(dt/R1/C1/C1*np.exp(-dt/R1/C1)*x[1][0]*sigma_C1-C1*dt/R1/C1/C1*np.exp(-dt/R1/C1)*current[k-1]*sigma_C1,2)+np.power(R1*(1-np.exp(-dt/R1/C1))*sigma_i,2)
            # Q3 = np.power(np.exp(-dt/R2/C2)*sigma_V,2)+np.power(dt/R2/R2/C2*np.exp(-dt/R2/C2)*x[2][0]*sigma_R2+1*(1-np.exp(-dt/R2/C2))*current[k-1]*sigma_R2-R2*dt/R2/R2/C2*np.exp(-dt/R2/C2)*current[k-1]*sigma_R2,2)+np.power(dt/R2/C2/C2*np.exp(-dt/R2/C2)*x[2][0]*sigma_C2-C2*dt/R2/C2/C2*np.exp(-dt/R2/C2)*current[k-1]*sigma_C2,2)+np.power(R2*(1-np.exp(-dt/R2/C2))*sigma_i,2)
            # not including sigma_V
            Q2 = np.power(dt/R1/R1/C1*np.exp(-dt/R1/C1)*x[1][0]*sigma_R1+1*(1-np.exp(-dt/R1/C1))*current[k-1]*sigma_R1-R1*dt/R1/R1/C1*np.exp(-dt/R1/C1)*current[k-1]*sigma_R1,2)+np.power(dt/R1/C1/C1*np.exp(-dt/R1/C1)*x[1][0]*sigma_C1-C1*dt/R1/C1/C1*np.exp(-dt/R1/C1)*current[k-1]*sigma_C1,2)+np.power(R1*(1-np.exp(-dt/R1/C1))*sigma_i,2)
            Q3 = np.power(dt/R2/R2/C2*np.exp(-dt/R2/C2)*x[2][0]*sigma_R2+1*(1-np.exp(-dt/R2/C2))*current[k-1]*sigma_R2-R2*dt/R2/R2/C2*np.exp(-dt/R2/C2)*current[k-1]*sigma_R2,2)+np.power(dt/R2/C2/C2*np.exp(-dt/R2/C2)*x[2][0]*sigma_C2-C2*dt/R2/C2/C2*np.exp(-dt/R2/C2)*current[k-1]*sigma_C2,2)+np.power(R2*(1-np.exp(-dt/R2/C2))*sigma_i,2)

            Q = np.array([[Q1,0,0],[0,Q3,0],[0,0,Q3]])
            R = np.array([np.power(sigma_V,2)])

        # print(B)
        x_f = np.matmul(F,x) + B*current[k-1]
        # if k == 5:
        #     print(B)
        #     print(current[k-1])
        #     print(np.matmul(B,np.array([current[k-1]])))
        #     print(B*current[k-1])
        #     print(np.matmul(F,x))
        F_T = F.T
        P_f = np.matmul(F,np.matmul(P,F_T)) + Q
        # if k == 5:
        #     print(np.matmul(F,np.matmul(P,F_T)))
        #     print(Q)
        #     print(np.matmul(F,np.matmul(P,F_T)) + Q)
        
        dOCV_dSOC = dOCV_dSOC_fn_here(x_f[0][0])
        H = np.array([[dOCV_dSOC,-1,-1]])
        H_T = H.T
        K1 = np.matmul(P_f,H_T)
        K2 = np.matmul(H,np.matmul(P_f,H_T)) + R
        if K2[0][0] == 0:
            K2 = 100
            print('Inverse of 0')
        else:
            K2 = 1/K2
        K = np.matmul(K1,K2)

        # if k == 5:
        #     print(H)
        #     print(H_T)
        #     print(np.matmul(P_f,H_T))
        #     print(np.matmul(H,np.matmul(P_f,H_T)) )
        #     print(R)
        #     print(np.matmul(H,np.matmul(P_f,H_T)) + R)
        #     print(K)
        
        z_k = voltage[k]
        y_k = OCV_fn_here(x_f[0][0]) - x_f[1][0] - x_f[2][0] - current[k-1]*R0

        x = x_f + np.matmul(K , np.array([(z_k - y_k)]))
        x[0, x[0] < 0] = 0
        x[0, x[0] > 1] = 1

        I = np.identity(3)
        P1 = (I-np.matmul(K,H))
        P = np.matmul(P1,P_f)

        SOC[k] = x[0][0]
        y[k] = y_k
        yocv[k] = OCV_fn_here(x_f[0][0])
        y0[k] = current[k-1]*R0
        y1[k] = x_f[1][0]
        y2[k] = x_f[2][0]
        K_track.append(K.copy())

    K_track = np.array(K_track)

    return [SOC,y,yocv,y0,y1,y2,K_track]

SOC_CoulombCounting = coulomb_counting(t_data,i_data,SOC_o,Qnom)
SOC_OCVSOCLookup = OCV_SOC_lookup(V_data,OCV,SOC)

T_array = np.array([0, 25, 45])
R0_bar = np.array([0.110276931, 0.060897589, 0.069535131])
R1_bar = np.array([0.013821696,0.009824284,0.008271106])
C1_bar = np.array([88.74387435,103.7457295,103.0754901])
R2_bar = np.array([0.038891322,0.003341077,0.003035031])
C2_bar = np.array([905.3248399,1018.15118,1005.334909])
R0_fn = interp1d(T_array,R0_bar,fill_value=(R0_bar[0],R0_bar[-1]))
R1_fn = interp1d(T_array,R1_bar,fill_value=(R1_bar[0],R1_bar[-1]))
C1_fn = interp1d(T_array,C1_bar,fill_value=(C1_bar[0],C1_bar[-1]))
R2_fn = interp1d(T_array,R2_bar,fill_value=(R2_bar[0],R2_bar[-1]))
C2_fn = interp1d(T_array,C2_bar,fill_value=(C2_bar[0],C2_bar[-1]))

R0 = R0_fn(T)
R1 = R1_fn(T)
C1 = C1_fn(T)
R2 = R2_fn(T)
C2 = C2_fn(T)

params = np.array([R0,R1,C1,R2,C2])
print('R0=',params[0])
print('R1=',params[1])
print('C1=',params[2])
print('R2=',params[3])
print('C2=',params[4])
sigma_SOC = 0.02 # 2 %
sigma_V = 0.05 # 50 mV
P = np.array([[np.power(sigma_SOC,2),0,0],[0,np.power(sigma_V,2),0],[0,0,np.power(sigma_V,2)]])
sigma_i = 0.01 # 0.01 A
sigma_Q = 0.02 # 0.2 Ah

sigma_R0_bar = np.array([0.01,0.01,0.01]) * R0_bar
sigma_R1_bar = np.array([0.01,0.01,0.01]) * R1_bar
sigma_C1_bar = np.array([0.01,0.01,0.01]) * C1_bar
sigma_R2_bar = np.array([0.01,0.01,0.01]) * R2_bar
sigma_C2_bar = np.array([0.01,0.01,0.01]) * C2_bar
sigma_R0_fn = interp1d(T_array,sigma_R0_bar,fill_value=(sigma_R0_bar[0],sigma_R0_bar[-1]))
sigma_R1_fn = interp1d(T_array,sigma_R1_bar,fill_value=(sigma_R1_bar[0],sigma_R1_bar[-1]))
sigma_C1_fn = interp1d(T_array,sigma_C1_bar,fill_value=(sigma_C1_bar[0],sigma_C1_bar[-1]))
sigma_R2_fn = interp1d(T_array,sigma_R2_bar,fill_value=(sigma_R2_bar[0],sigma_R2_bar[-1]))
sigma_C2_fn = interp1d(T_array,sigma_C2_bar,fill_value=(sigma_C2_bar[0],sigma_C2_bar[-1]))

sigma_R0 = sigma_R0_fn(T)
sigma_R1 = sigma_R1_fn(T)
sigma_C1 = sigma_C1_fn(T)
sigma_R2 = sigma_R2_fn(T)
sigma_C2 = sigma_C2_fn(T)

sigma_bar = np.array([sigma_SOC,sigma_V,sigma_i,sigma_Q,sigma_R0,sigma_R1,sigma_C1,sigma_R2,sigma_C2])
# add sigma_R1, sigma_R2, sigma_C1
# also think about adding back V0 as a state (yeah F will be zero)
# will def work I think but do it as SOC_OnlineCalcSim_withR0State.py pleases

method_flag = 1
result_KalmanFilter = kalman_fitler(t_data,i_data,V_data,OCV,dOCV_dSOC,SOC,Qnom,SOC_o,P,sigma_bar,params,method_flag)
# print(SOC_KalmanFilter)
SOC_KalmanFilter = result_KalmanFilter[0]
V_KalmanFilter = result_KalmanFilter[1]
V_KalmanFilter_ocv = result_KalmanFilter[2]
V_KalmanFilter_0 = result_KalmanFilter[3]
V_KalmanFilter_1 = result_KalmanFilter[4]
V_KalmanFilter_2 = result_KalmanFilter[5]
K_KalmanFilter = result_KalmanFilter[6]
K_KalmanFilter_1 = K_KalmanFilter[:,0,0]
K_KalmanFilter_2 = K_KalmanFilter[:,1,0]
K_KalmanFilter_3 = K_KalmanFilter[:,2,0]

residuals_CoulombCounting = (SOC_CoulombCounting - SOC_data)
SS_res_CoulombCounting = np.sum(residuals_CoulombCounting**2)
SS_tot = np.sum((SOC_data-np.mean(SOC_data))**2)
R2_fit_CoulombCounting = 1-SS_res_CoulombCounting/SS_tot # this is where the error is coming from
RMS_CoulombCounting = SS_res_CoulombCounting * 1/np.sqrt(len(SOC_data))

residuals_KalmanFilter = (SOC_KalmanFilter - SOC_data)
SS_res_KalmanFilter = np.sum(residuals_KalmanFilter**2)
# SS_tot = np.sum((SOC_data-np.mean(SOC_data))**2)
R2_fit_KalmanFilter = 1-SS_res_KalmanFilter/SS_tot # this is where the error is coming from
RMS_KalmanFilter = SS_res_KalmanFilter * 1/np.sqrt(len(SOC_data))

print('Coulomb Counting RMS: ', RMS_CoulombCounting)
print('Kalman Filter RMS: ', RMS_KalmanFilter)

# voltage plots
fig, axs = plt.subplots(5,1,figsize=(8, 5), sharex=True)

axs[0].plot(t_data, V_data, marker='o', markersize=size_for_markers_plot,label='Actual V')
axs[0].plot(t_data, V_KalmanFilter, marker='o', markersize=size_for_markers_plot,label='Model V')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('V')
axs[0].set_title("Terminal Voltage")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(t_data, V_KalmanFilter_ocv, marker='o', markersize=size_for_markers_plot,label='Model OCV')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('OCV')
axs[1].set_title("OCV")
axs[1].legend()
axs[1].grid(True)

axs[2].plot(t_data, V_KalmanFilter_0, marker='o', markersize=size_for_markers_plot,label='Model V0')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('V0')
axs[2].set_title("V0")
axs[2].legend()
axs[2].grid(True)

axs[3].plot(t_data, V_KalmanFilter_1, marker='o', markersize=size_for_markers_plot,label='Model V1')
axs[3].set_xlabel('Time (s)')
axs[3].set_ylabel('V1')
axs[3].set_title("V1")
axs[3].legend()
axs[3].grid(True)

axs[4].plot(t_data, V_KalmanFilter_2, marker='o', markersize=size_for_markers_plot,label='Model V2')
axs[4].set_xlabel('Time (s)')
axs[4].set_ylabel('V2')
axs[4].set_title("V2")
axs[4].legend()
axs[4].grid(True)

# Kalman gain plots
fig, axs = plt.subplots(3,1,figsize=(8, 5), sharex=True)
axs[0].plot(t_data, K_KalmanFilter_1, marker='o', markersize=size_for_markers_plot,label='Actual SOC')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('K1')
axs[0].set_title("Kalman Gain vs Time")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(t_data, K_KalmanFilter_2, marker='o', markersize=size_for_markers_plot,label='Actual SOC')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('K2')
# axs[1].set_title("Kalman Gain vs Time")
axs[1].legend()
axs[1].grid(True)

axs[2].plot(t_data, K_KalmanFilter_3, marker='o', markersize=size_for_markers_plot,label='Actual SOC')
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('K3')
# axs[2].set_title("Kalman Gain vs Time")
axs[2].legend()
axs[2].grid(True)

# error (z_k-y_k)
fig, axs = plt.subplots(2,1,figsize=(8, 5), sharex=True)

axs[0].plot(t_data,V_data-V_KalmanFilter,marker='o', markersize=size_for_markers_plot,label='Error')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('z_k - y_k')
axs[0].set_title("z_k - y_k")
axs[0].legend()
axs[0].grid(True)

axs[1].plot(t_data,K_KalmanFilter_1*(V_data-V_KalmanFilter),marker='o', markersize=size_for_markers_plot,label='K*error')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('K * (z_k - y_k)')
axs[1].set_title("K * (z_k - y_k)")
axs[1].legend()
axs[1].grid(True)

# SOC Comparison PLOTS
fig, axs = plt.subplots(figsize=(8, 5), sharex=True)
axs.plot(t_data,SOC_discharge,marker='o', markersize=size_for_markers_plot,label='Actual SOC')
axs.plot(t_data,SOC_CoulombCounting,marker='o', markersize=size_for_markers_plot,label='Coulomb Counting')
# axs.plot(t_data,SOC_OCVSOCLookup,marker='o', markersize=size_for_markers_plot,label='OCV SOC Lookup')
axs.plot(t_data,SOC_KalmanFilter,marker='o', markersize=size_for_markers_plot,label='Kalman Filter')
axs.set_xlabel('Time (s)')
axs.set_ylabel('SOC')
axs.legend()
axs.grid(True)

fig, axs = plt.subplots(figsize=(8, 5), sharex=True)
axs.plot(t_data,V_data,marker='o', markersize=size_for_markers_plot,label='Actual V')
axs.plot(t_data,V_KalmanFilter,marker='o', markersize=size_for_markers_plot,label='Kalman Filter')
axs.set_xlabel('Time (s)')
axs.set_ylabel('V (V)')
axs.legend()
axs.grid(True)

plt.show()