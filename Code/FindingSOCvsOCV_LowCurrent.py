import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

Qnom = 2000 # mAh
Qnom = Qnom/1000 # Ah
SOC_common = np.linspace(0,1,101)

# use Qnom flag
Qnom_flag = 0

# Low Current OCV - Sample 1 - 0 deg C
df1 = pd.read_excel('../Data/02_24_2016_SP20-1_0C_lowcurrentOCV.xls', sheet_name='Channel_1-005_1')
df2 = pd.read_excel('../Data/02_24_2016_SP20-1_0C_lowcurrentOCV.xls', sheet_name='Channel_1-005_2')
df_sam1_0degC = pd.concat([df1, df2], ignore_index=True)

# Low Current OCV - Sample 2 - 25 deg C
df1 = pd.read_excel('../Data/11_16_2015_low current OCV test_SP20-3.xls', sheet_name='Channel_1-004_1')
df2 = pd.read_excel('../Data/11_16_2015_low current OCV test_SP20-3.xls', sheet_name='Channel_1-004_2')
df_sam2_25degC = pd.concat([df1, df2], ignore_index=True)

# Low Current OCV - Sample 2 - 0 deg C
df1 = pd.read_excel('../Data/03_03_2016_SP20-3_0C_lowcurrentOCV.xls', sheet_name='Channel_1-005_1')
df2 = pd.read_excel('../Data/03_03_2016_SP20-3_0C_lowcurrentOCV.xls', sheet_name='Channel_1-005_2')
df3 = pd.read_excel('../Data/03_03_2016_SP20-3_0C_lowcurrentOCV.xls',sheet_name='Channel_1-005_3')
df_sam2_0degC = pd.concat([df1, df2, df3], ignore_index=True)

# Plotting
for i in range(3):
    if i == 0:
        data = df_sam1_0degC
    elif i == 1:
        data = df_sam2_25degC
    elif i == 2:
        data = df_sam2_0degC

    TestTime = data['Test_Time(s)']
    Voltage = data['Voltage(V)']
    dV_dt = data['dV/dt(V/s)']
    Current = data['Current(A)']
    StepIndex = data['Step_Index']
    CycleIndex = data['Cycle_Index']
    ChargeCapacity = data['Charge_Capacity(Ah)']
    DischargeCapacity = data['Discharge_Capacity(Ah)']


    fig, axs = plt.subplots(4, 1, figsize=(8, 5), sharex=True)
    size_for_markers_plot = 2
    size_for_markers_scatter = 22

    axs[0].plot(TestTime, Voltage, marker='o', markersize=size_for_markers_plot)
    axs[0].set_title(f'All data')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Voltage (V)')
    axs[0].grid(True)

    # axs[1].plot(TestTime, dV_dt, marker='o', markersize=size_for_markers_plot)
    # axs[1].set_xlabel('Time')
    # axs[1].set_ylabel('dV/dt (V/s)')
    # axs[1].grid(True)

    axs[1].plot(TestTime, ChargeCapacity, marker='o', label='Charge Capacity', markersize=size_for_markers_plot)
    axs[1].plot(TestTime, DischargeCapacity, marker='o', label='Discharge Capacity', markersize=size_for_markers_plot)
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Charge Capacity (Ah)')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(TestTime, Current, marker='o', markersize=size_for_markers_plot)
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Current (A)')
    axs[2].grid(True)

    axs[3].scatter(TestTime, StepIndex, label='Step Index', marker='o', s=size_for_markers_scatter)
    axs[3].scatter(TestTime, CycleIndex, label='Cycle Index', marker='o', s=size_for_markers_scatter)
    y_max = max(max(StepIndex),max(CycleIndex))
    axs[3].set_yticks(range(0, int(y_max)+1))
    axs[3].set_xlabel('Time')
    axs[3].set_ylabel('Index')
    axs[3].legend()
    axs[3].grid(True)


# OCV steps used for OCV and SOC relationship
OCV_steps = [6,8]

# Low Current OCV - Sample 1 - 0 deg C
discharge_SOC_data_sam1_0degC = df_sam1_0degC[df_sam1_0degC['Step_Index'].isin([OCV_steps[0]])]
charge_SOC_data_sam1_0degC = df_sam1_0degC[df_sam1_0degC['Step_Index'].isin([OCV_steps[1]])]

Qmax = max(discharge_SOC_data_sam1_0degC['Discharge_Capacity(Ah)'])
Qmin = min(discharge_SOC_data_sam1_0degC['Discharge_Capacity(Ah)'])
if Qnom_flag == 1:
    Qmax = Qnom
    Qmin = 0
print('Sample 1: 0 deg C Data')
print('Discharge segment')
print('Qmax: ', Qmax, ' Ah')
print('Qmin: ', Qmin, ' Ah')
discharge_V_sam1_0degC = discharge_SOC_data_sam1_0degC['Voltage(V)']
discharge_SOC_sam1_0degC = (Qmax - discharge_SOC_data_sam1_0degC['Discharge_Capacity(Ah)'])/(Qmax-Qmin)

Qmax = max(charge_SOC_data_sam1_0degC['Charge_Capacity(Ah)'])
Qmin = min(charge_SOC_data_sam1_0degC['Charge_Capacity(Ah)'])
if Qnom_flag == 1:
    Qmax = Qnom
    Qmin = 0
print('Charge segment')
print('Qmax: ', Qmax, ' Ah')
print('Qmin: ', Qmin, ' Ah')
charge_V_sam1_0degC = charge_SOC_data_sam1_0degC['Voltage(V)']
charge_SOC_sam1_0degC = (charge_SOC_data_sam1_0degC['Charge_Capacity(Ah)']-Qmin)/(Qmax-Qmin)

# Assume: soc1, ocv1 and soc2, ocv2 are your input data
interp1 = interp1d(discharge_SOC_sam1_0degC, discharge_V_sam1_0degC, kind='linear', bounds_error=False, fill_value='extrapolate')
interp2 = interp1d(charge_SOC_sam1_0degC, charge_V_sam1_0degC, kind='linear', bounds_error=False, fill_value='extrapolate')

ocv1_interp = interp1(SOC_common)
ocv2_interp = interp2(SOC_common)

OCV_sam1_0degC = (ocv1_interp + ocv2_interp) / 2
dOCV_sam1_0degC_dSOC = np.gradient(OCV_sam1_0degC, SOC_common)


# Low Current OCV - Sample 2 - 0 deg C
discharge_SOC_data_sam2_0degC = df_sam2_0degC[df_sam2_0degC['Step_Index'].isin([OCV_steps[0]])]
charge_SOC_data_sam2_0degC = df_sam2_0degC[df_sam2_0degC['Step_Index'].isin([OCV_steps[1]])]

Qmax = max(discharge_SOC_data_sam2_0degC['Discharge_Capacity(Ah)'])
Qmin = min(discharge_SOC_data_sam2_0degC['Discharge_Capacity(Ah)'])
if Qnom_flag == 1:
    Qmax = Qnom
    Qmin = 0
print('Sample 2: 0 deg C Data')
print('Discharge segment')
print('Qmax: ', Qmax, ' Ah')
print('Qmin: ', Qmin, ' Ah')
discharge_V_sam2_0degC = discharge_SOC_data_sam2_0degC['Voltage(V)']
discharge_SOC_sam2_0degC = (Qmax - discharge_SOC_data_sam2_0degC['Discharge_Capacity(Ah)'])/(Qmax-Qmin)

Qmax = max(charge_SOC_data_sam2_0degC['Charge_Capacity(Ah)'])
Qmin = min(charge_SOC_data_sam2_0degC['Charge_Capacity(Ah)'])
if Qnom_flag == 1:
    Qmax = Qnom
    Qmin = 0
print('Charge segment')
print('Qmax: ', Qmax, ' Ah')
print('Qmin: ', Qmin, ' Ah')
charge_V_sam2_0degC = charge_SOC_data_sam2_0degC['Voltage(V)']
charge_SOC_sam2_0degC = (charge_SOC_data_sam2_0degC['Charge_Capacity(Ah)']-Qmin)/(Qmax-Qmin)

# Assume: soc1, ocv1 and soc2, ocv2 are your input data
interp1 = interp1d(discharge_SOC_sam2_0degC, discharge_V_sam2_0degC, kind='linear', bounds_error=False, fill_value='extrapolate')
interp2 = interp1d(charge_SOC_sam2_0degC, charge_V_sam2_0degC, kind='linear', bounds_error=False, fill_value='extrapolate')

ocv1_interp = interp1(SOC_common)
ocv2_interp = interp2(SOC_common)

OCV_sam2_0degC = (ocv1_interp + ocv2_interp) / 2
dOCV_sam2_0degC_dSOC = np.gradient(OCV_sam2_0degC, SOC_common)


# Low Current OCV - Sample 2 - 25 deg C
discharge_SOC_data_sam2_25degC = df_sam2_25degC[df_sam2_25degC['Step_Index'].isin([OCV_steps[0]])]
charge_SOC_data_sam2_25degC = df_sam2_25degC[df_sam2_25degC['Step_Index'].isin([OCV_steps[1]])]

Qmax = max(discharge_SOC_data_sam2_25degC['Discharge_Capacity(Ah)'])
Qmin = min(discharge_SOC_data_sam2_25degC['Discharge_Capacity(Ah)'])
if Qnom_flag == 1:
    Qmax = Qnom
    Qmin = 0
print('Sample 2: 25 deg C Data')
print('Discharge segment')
print('Qmax: ', Qmax, ' Ah')
print('Qmin: ', Qmin, ' Ah')
discharge_V_sam2_25degC = discharge_SOC_data_sam2_25degC['Voltage(V)']
discharge_SOC_sam2_25degC = (Qmax - discharge_SOC_data_sam2_25degC['Discharge_Capacity(Ah)'])/(Qmax-Qmin)

Qmax = max(charge_SOC_data_sam2_25degC['Charge_Capacity(Ah)'])
Qmin = min(charge_SOC_data_sam2_25degC['Charge_Capacity(Ah)'])
if Qnom_flag == 1:
    Qmax = Qnom
    Qmin = 0
print('Charge segment')
print('Qmax: ', Qmax, ' Ah')
print('Qmin: ', Qmin, ' Ah')
charge_V_sam2_25degC = charge_SOC_data_sam2_25degC['Voltage(V)']
charge_SOC_sam2_25degC = (charge_SOC_data_sam2_25degC['Charge_Capacity(Ah)']-Qmin)/(Qmax-Qmin)

# Assume: soc1, ocv1 and soc2, ocv2 are your input data
interp1 = interp1d(discharge_SOC_sam2_25degC, discharge_V_sam2_25degC, kind='linear', bounds_error=False, fill_value='extrapolate')
interp2 = interp1d(charge_SOC_sam2_25degC, charge_V_sam2_25degC, kind='linear', bounds_error=False, fill_value='extrapolate')

ocv1_interp = interp1(SOC_common)
ocv2_interp = interp2(SOC_common)

OCV_sam2_25degC = (ocv1_interp + ocv2_interp) / 2
dOCV_sam2_25degC_dSOC = np.gradient(OCV_sam2_25degC, SOC_common)


####
# OCV = (OCV_sam1_0degC+OCV_sam2_25degC)/2
OCV = (OCV_sam1_0degC+OCV_sam2_0degC+OCV_sam2_25degC)/3
SOC = SOC_common
dOCV_dSOC = np.gradient(OCV, SOC)

# export data
np.savez('OCV_dOCV_dSOC_SOC_LowCurrent', v1=OCV, v2=dOCV_dSOC, v3=SOC)
np.savez('OCV_dOCV_dSOC_SOC_LowCurrent_sam1_0degC', v1=OCV_sam1_0degC, v2=dOCV_sam1_0degC_dSOC, v3=SOC)
np.savez('OCV_dOCV_dSOC_SOC_LowCurrent_sam2_0degC', v1=OCV_sam2_0degC, v2=dOCV_sam2_0degC_dSOC, v3=SOC)
np.savez('OCV_dOCV_dSOC_SOC_LowCurrent_sam2_25degC', v1=OCV_sam2_25degC, v2=dOCV_sam2_25degC_dSOC, v3=SOC)

# export data to csv
data_to_export = {
    'SOC': SOC_common,
    'OCV_sam1_0degC' : OCV_sam1_0degC,
    'OCV_sam2_0degC' : OCV_sam2_0degC,
    'OCV_sam2_25degC' : OCV_sam2_25degC
}
df_export = pd.DataFrame(data_to_export)
df_export.to_csv('LowCurrentOCVvsSOC.csv', index=False)

# OCV = OCV 
# dOCV_dSOC = dOCV_dSOC
# OCV = OCV_sam1_0degC 
# dOCV_dSOC = dOCV_sam1_0degC_dSOC

# # trim indices if we want to 
# # SOC_trim_indices = np.where((SOC >= 0.1) & (SOC <= 0.9))[0]  # Indices to keep
# SOC_trim_indices = np.where((SOC >= 0.1))[0]  # Indices to keep
# OCV = OCV[SOC_trim_indices]
# dOCV_dSOC = dOCV_dSOC[SOC_trim_indices]
# SOC_common = SOC_common[SOC_trim_indices]

# plotting
fig, axs = plt.subplots(2,1,figsize=(8, 5), sharex=True)
# axs[0].plot(SOC_common,OCV,label='OCV vs SOC',color='blue')
axs[0].plot(SOC_common,OCV,label='Sam1 0degC',color='blue')
# Now show all at once
# axs[0].set_xlabel('SOC')
axs[0].set_ylabel('OCV (V)')
axs[0].set_xlim(0,1)
axs[0].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
axs[0].grid(True)
axs[0].set_title(f'Low Current OCV and dOCV/dSOC vs SOC')
axs[0].legend()

axs[1].plot(SOC_common,dOCV_dSOC,color='blue',label='d(OCV)/d(SOC)')
axs[1].set_xlabel('SOC')
axs[1].set_ylabel('dOCV/dSOC (V)')
axs[1].set_xlim(0,1)
axs[1].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
axs[1].grid(True)

# plotting
fig, axs = plt.subplots(figsize=(8, 5), sharex=True)
axs.plot(discharge_SOC_sam1_0degC,discharge_V_sam1_0degC,linestyle='--',color='blue',label='Discharge Sam1 0degC')
axs.plot(charge_SOC_sam1_0degC,charge_V_sam1_0degC,linestyle='--',color='black',label='Charge Sam1 0degC')
# axs.plot(discharge_SOC_sam2_0degC,discharge_V_sam2_0degC,linestyle='--',color='green',label='Discharge Sam2 0degC')
# axs.plot(charge_SOC_sam2_0degC,charge_V_sam2_0degC,linestyle='--',color='yellow',label='Charge Sam2 0degC')
# axs.plot(discharge_SOC_sam2_25degC,discharge_V_sam2_25degC,linestyle='--',color='red',label='Discharge Sam2 25degC')
# axs.plot(charge_SOC_sam2_25degC,charge_V_sam2_25degC,linestyle='--',color='orange',label='Charge Sam2 25degC')
axs.plot(SOC_common,OCV_sam1_0degC,label='Sam1 0degC',color='blue')
# axs.plot(SOC_common,OCV_sam2_0degC,label='Sam2 0degC',color='green')
# axs.plot(SOC_common,OCV_sam2_25degC,label='Sam2 25degC',color='red')

# Now show all at once
plt.xlabel('SOC')
plt.ylabel('OCV (V)')
plt.title('Low Current OCV vs SOC')
plt.xlim(0,1)
plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.legend()

plt.show()