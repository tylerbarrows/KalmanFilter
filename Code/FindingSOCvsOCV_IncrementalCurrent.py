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

# Incremental Current OCV - Sample 1 - Data for 0 deg C
df1 = pd.read_excel('../Data/02_26_2016_SP20-1_0C_incrementalOCV.xls',sheet_name='Channel_1-005_1')
df2 = pd.read_excel('../Data/02_26_2016_SP20-1_0C_incrementalOCV.xls',sheet_name='Channel_1-005_2')
df3 = pd.read_excel('../Data/02_26_2016_SP20-1_0C_incrementalOCV.xls',sheet_name='Channel_1-005_3')
data_sam1_0degC = pd.concat([df1, df2, df3], ignore_index=True)

# Incremental Current OCV - Sample 1 - Data for 25 deg C
df1 = pd.read_excel('../Data/12_2_2015_Incremental OCV test_SP20-1.xlsx',sheet_name='Channel_1-005_1')
df2 = pd.read_excel('../Data/12_2_2015_Incremental OCV test_SP20-1.xlsx',sheet_name='Channel_1-005_2')
df3 = pd.read_excel('../Data/12_2_2015_Incremental OCV test_SP20-1.xlsx',sheet_name='Channel_1-005_3')
data_sam1_25degC = pd.concat([df1, df2, df3], ignore_index=True)

# Incremental Current OCV - Sample 1 - Data for 45 deg C
df1 = pd.read_excel('../Data/12_09_2015_Incremental OCV test_SP20-1.xlsx',sheet_name='Channel_1-005_1')
df2 = pd.read_excel('../Data/12_09_2015_Incremental OCV test_SP20-1.xlsx',sheet_name='Channel_1-005_2')
df3 = pd.read_excel('../Data/12_09_2015_Incremental OCV test_SP20-1.xlsx',sheet_name='Channel_1-005_3')
data_sam1_45degC = pd.concat([df1, df2, df3], ignore_index=True)

# Incremental Current OCV - Sample 2 - Data for 0 deg C
df1 = pd.read_excel('../Data/03_09_2016_SP20-3_0C_incrementalOCV.xls',sheet_name='Channel_1-005_1')
df2 = pd.read_excel('../Data/03_09_2016_SP20-3_0C_incrementalOCV.xls',sheet_name='Channel_1-005_2')
df3 = pd.read_excel('../Data/03_09_2016_SP20-3_0C_incrementalOCV.xls',sheet_name='Channel_1-005_3')
data_sam2_0degC = pd.concat([df1, df2, df3], ignore_index=True)

# Incremental Current OCV - Sample 2 - Data for 25 deg C
df1 = pd.read_excel('../Data/12_2_2015_Incremental OCV test_SP20-3.xlsx',sheet_name='Channel_1-006_1')
df2 = pd.read_excel('../Data/12_2_2015_Incremental OCV test_SP20-3.xlsx',sheet_name='Channel_1-006_2')
df3 = pd.read_excel('../Data/12_2_2015_Incremental OCV test_SP20-3.xlsx',sheet_name='Channel_1-006_3')
data_sam2_25degC = pd.concat([df1, df2, df3], ignore_index=True)

# Incremental Current OCV - Sample 2 - Data for 45 deg C
df1 = pd.read_excel('../Data/12_09_2015_Incremental OCV test_SP20-3.xlsx',sheet_name='Channel_1-006_1')
df2 = pd.read_excel('../Data/12_09_2015_Incremental OCV test_SP20-3.xlsx',sheet_name='Channel_1-006_2')
df3 = pd.read_excel('../Data/12_09_2015_Incremental OCV test_SP20-3.xlsx',sheet_name='Channel_1-006_3')
data_sam2_45degC = pd.concat([df1, df2, df3], ignore_index=True)


# Incremental Current OCV - Sample 1 - Data for 0 deg C
data = data_sam1_0degC
# cycle and step indices to use for SOC
SOC_cycle_indices_discharge = [1,2,3,4,5,6,7,8,9,10]
SOC_step_indices_discharge = [4,6,6,6,6,6,6,6,6,6]
SOC_cycle_indices_charge = [10,11,12,13,14,15,16,17,18,19,19] # could add ten to the beginning
SOC_step_indices_charge = [6,10,10,10,10,10,10,10,10,13,16] # could add six to the beginning

#
TestTime = data['Test_Time(s)']
Voltage = data['Voltage(V)']
dV_dt = data['dV/dt(V/s)']
Current = data['Current(A)']
StepIndex = data['Step_Index']
CycleIndex = data['Cycle_Index']
ChargeCapacity = data['Charge_Capacity(Ah)']
DischargeCapacity = data['Discharge_Capacity(Ah)']

# OCV and SOC arrays
size_discharge = len(SOC_cycle_indices_discharge)
discharge_OCV = np.empty(size_discharge)
discharge_SOC = np.empty(size_discharge)
size_charge = len(SOC_cycle_indices_charge)
charge_OCV = np.empty(size_charge)
charge_SOC = np.empty(size_charge)

# Discharge
Qmax = max(DischargeCapacity)
Qmin = min(DischargeCapacity)
if Qnom_flag == 1:
    Qmax = Qnom
    Qmin = 0
for i in range(size_discharge):
    data_to_use = data[(data['Cycle_Index'] == SOC_cycle_indices_discharge[i]) & (data['Step_Index'] == SOC_step_indices_discharge[i])]
    discharge_OCV[i] = data_to_use['Voltage(V)'].iloc[-1]
    discharge_SOC[i] = (Qmax-data_to_use['Discharge_Capacity(Ah)'].iloc[-1])/(Qmax-Qmin)

# Charge
Qmax = max(ChargeCapacity)
Qmin = min(ChargeCapacity)
if Qnom_flag == 1:
    Qmax = Qnom
    Qmin = 0
for i in range(size_charge):
    data_to_use = data[(data['Cycle_Index'] == SOC_cycle_indices_charge[i]) & (data['Step_Index'] == SOC_step_indices_charge[i])]
    charge_OCV[i] = data_to_use['Voltage(V)'].iloc[-1]
    charge_SOC[i] = (data_to_use['Charge_Capacity(Ah)'].iloc[-1]-Qmin)/(Qmax-Qmin)

# Assume: soc1, ocv1 and soc2, ocv2 are your input data
interp1 = interp1d(discharge_SOC, discharge_OCV, kind='cubic', bounds_error=False, fill_value='extrapolate')
interp2 = interp1d(charge_SOC, charge_OCV, kind='cubic', bounds_error=False, fill_value='extrapolate')

ocv1_interp = interp1(SOC_common)
ocv2_interp = interp2(SOC_common)

OCV_sam1_0degC = (ocv1_interp+ocv2_interp)/2
dOCV_dSOC_sam1_0degC = np.gradient(OCV_sam1_0degC, SOC_common)
OCV_sam1_0degC_discharge = ocv1_interp
OCV_sam1_0degC_charge = ocv2_interp


# Incremental Current OCV - Sample 1 - Data for 25 deg C
data = data_sam1_25degC
# cycle and step indices to use for SOC
SOC_cycle_indices_discharge = [1,2,3,4,5,6,7,8,9,10]
SOC_step_indices_discharge = [4,6,6,6,6,6,6,6,6,8]
SOC_cycle_indices_charge = [10,10,11,12,13,14,15,16,17] # could add ten to the beginning
SOC_step_indices_charge = [8,10,10,10,10,10,10,10,10] # could add six to the beginning

#
TestTime = data['Test_Time(s)']
Voltage = data['Voltage(V)']
dV_dt = data['dV/dt(V/s)']
Current = data['Current(A)']
StepIndex = data['Step_Index']
CycleIndex = data['Cycle_Index']
ChargeCapacity = data['Charge_Capacity(Ah)']
DischargeCapacity = data['Discharge_Capacity(Ah)']

# OCV and SOC arrays
size_discharge = len(SOC_cycle_indices_discharge)
discharge_OCV = np.empty(size_discharge)
discharge_SOC = np.empty(size_discharge)
size_charge = len(SOC_cycle_indices_charge)
charge_OCV = np.empty(size_charge)
charge_SOC = np.empty(size_charge)

# Discharge
Qmax = max(DischargeCapacity)
Qmin = min(DischargeCapacity)
if Qnom_flag == 1:
    Qmax = Qnom
    Qmin = 0
for i in range(size_discharge):
    data_to_use = data[(data['Cycle_Index'] == SOC_cycle_indices_discharge[i]) & (data['Step_Index'] == SOC_step_indices_discharge[i])]
    discharge_OCV[i] = data_to_use['Voltage(V)'].iloc[-1]
    discharge_SOC[i] = (Qmax-data_to_use['Discharge_Capacity(Ah)'].iloc[-1])/(Qmax-Qmin)

# Charge
Qmax = max(ChargeCapacity)
Qmin = min(ChargeCapacity)
if Qnom_flag == 1:
    Qmax = Qnom
    Qmin = 0
for i in range(size_charge):
    data_to_use = data[(data['Cycle_Index'] == SOC_cycle_indices_charge[i]) & (data['Step_Index'] == SOC_step_indices_charge[i])]
    charge_OCV[i] = data_to_use['Voltage(V)'].iloc[-1]
    charge_SOC[i] = (data_to_use['Charge_Capacity(Ah)'].iloc[-1]-Qmin)/(Qmax-Qmin)

# Assume: soc1, ocv1 and soc2, ocv2 are your input data
interp1 = interp1d(discharge_SOC, discharge_OCV, kind='cubic', bounds_error=False, fill_value='extrapolate')
interp2 = interp1d(charge_SOC, charge_OCV, kind='cubic', bounds_error=False, fill_value='extrapolate')

ocv1_interp = interp1(SOC_common)
ocv2_interp = interp2(SOC_common)

OCV_sam1_25degC = (ocv1_interp+ocv2_interp)/2
dOCV_dSOC_sam1_25degC = np.gradient(OCV_sam1_25degC, SOC_common)


# Incremental Current OCV - Sample 1 - Data for 45 deg C
data = data_sam1_45degC
# cycle and step indices to use for SOC
SOC_cycle_indices_discharge = [1,1,2,3,4,5,6,7,8,9,10]
SOC_step_indices_discharge = [4,6,6,6,6,6,6,6,6,6,6]
SOC_cycle_indices_charge = [10,11,12,13,14,15,16,17,18,19,19] # could add ten to the beginning
SOC_step_indices_charge = [6,10,10,10,10,10,10,10,10,13,16] # could add six to the beginning

#
TestTime = data['Test_Time(s)']
Voltage = data['Voltage(V)']
dV_dt = data['dV/dt(V/s)']
Current = data['Current(A)']
StepIndex = data['Step_Index']
CycleIndex = data['Cycle_Index']
ChargeCapacity = data['Charge_Capacity(Ah)']
DischargeCapacity = data['Discharge_Capacity(Ah)']

# OCV and SOC arrays
size_discharge = len(SOC_cycle_indices_discharge)
discharge_OCV = np.empty(size_discharge)
discharge_SOC = np.empty(size_discharge)
size_charge = len(SOC_cycle_indices_charge)
charge_OCV = np.empty(size_charge)
charge_SOC = np.empty(size_charge)

# Discharge
Qmax = max(DischargeCapacity)
Qmin = min(DischargeCapacity)
if Qnom_flag == 1:
    Qmax = Qnom
    Qmin = 0
for i in range(size_discharge):
    data_to_use = data[(data['Cycle_Index'] == SOC_cycle_indices_discharge[i]) & (data['Step_Index'] == SOC_step_indices_discharge[i])]
    discharge_OCV[i] = data_to_use['Voltage(V)'].iloc[-1]
    discharge_SOC[i] = (Qmax-data_to_use['Discharge_Capacity(Ah)'].iloc[-1])/(Qmax-Qmin)

# Charge
Qmax = max(ChargeCapacity)
Qmin = min(ChargeCapacity)
if Qnom_flag == 1:
    Qmax = Qnom
    Qmin = 0
for i in range(size_charge):
    data_to_use = data[(data['Cycle_Index'] == SOC_cycle_indices_charge[i]) & (data['Step_Index'] == SOC_step_indices_charge[i])]
    charge_OCV[i] = data_to_use['Voltage(V)'].iloc[-1]
    charge_SOC[i] = (data_to_use['Charge_Capacity(Ah)'].iloc[-1]-Qmin)/(Qmax-Qmin)

# Assume: soc1, ocv1 and soc2, ocv2 are your input data
interp1 = interp1d(discharge_SOC, discharge_OCV, kind='cubic', bounds_error=False, fill_value='extrapolate')
interp2 = interp1d(charge_SOC, charge_OCV, kind='cubic', bounds_error=False, fill_value='extrapolate')

ocv1_interp = interp1(SOC_common)
ocv2_interp = interp2(SOC_common)

OCV_sam1_45degC = (ocv1_interp+ocv2_interp)/2
dOCV_dSOC_sam1_45degC = np.gradient(OCV_sam1_45degC, SOC_common)


# Incremental Current OCV - Sample 2 - Data for 0 deg C
data = data_sam2_0degC
# cycle and step indices to use for SOC
SOC_cycle_indices_discharge = [1,1,2,3,4,5,6,7,8,9,10]
SOC_step_indices_discharge = [4,6,6,6,6,6,6,6,6,6,8]
SOC_cycle_indices_charge = [10,10,11,12,13,14,15,16,17,18,18] # could add ten to the beginning
SOC_step_indices_charge = [8,10,10,10,10,10,10,10,10,13,16] # could add six to the beginning

#
TestTime = data['Test_Time(s)']
Voltage = data['Voltage(V)']
dV_dt = data['dV/dt(V/s)']
Current = data['Current(A)']
StepIndex = data['Step_Index']
CycleIndex = data['Cycle_Index']
ChargeCapacity = data['Charge_Capacity(Ah)']
DischargeCapacity = data['Discharge_Capacity(Ah)']

# OCV and SOC arrays
size_discharge = len(SOC_cycle_indices_discharge)
discharge_OCV = np.empty(size_discharge)
discharge_SOC = np.empty(size_discharge)
size_charge = len(SOC_cycle_indices_charge)
charge_OCV = np.empty(size_charge)
charge_SOC = np.empty(size_charge)

# Discharge
Qmax = max(DischargeCapacity)
Qmin = min(DischargeCapacity)
if Qnom_flag == 1:
    Qmax = Qnom
    Qmin = 0
for i in range(size_discharge):
    data_to_use = data[(data['Cycle_Index'] == SOC_cycle_indices_discharge[i]) & (data['Step_Index'] == SOC_step_indices_discharge[i])]
    discharge_OCV[i] = data_to_use['Voltage(V)'].iloc[-1]
    discharge_SOC[i] = (Qmax-data_to_use['Discharge_Capacity(Ah)'].iloc[-1])/(Qmax-Qmin)

# Charge
Qmax = max(ChargeCapacity)
Qmin = min(ChargeCapacity)
if Qnom_flag == 1:
    Qmax = Qnom
    Qmin = 0
for i in range(size_charge):
    data_to_use = data[(data['Cycle_Index'] == SOC_cycle_indices_charge[i]) & (data['Step_Index'] == SOC_step_indices_charge[i])]
    charge_OCV[i] = data_to_use['Voltage(V)'].iloc[-1]
    charge_SOC[i] = (data_to_use['Charge_Capacity(Ah)'].iloc[-1]-Qmin)/(Qmax-Qmin)

# Assume: soc1, ocv1 and soc2, ocv2 are your input data
interp1 = interp1d(discharge_SOC, discharge_OCV, kind='cubic', bounds_error=False, fill_value='extrapolate')
interp2 = interp1d(charge_SOC, charge_OCV, kind='cubic', bounds_error=False, fill_value='extrapolate')

ocv1_interp = interp1(SOC_common)
ocv2_interp = interp2(SOC_common)

OCV_sam2_0degC = (ocv1_interp+ocv2_interp)/2
dOCV_dSOC_sam2_0degC = np.gradient(OCV_sam2_0degC, SOC_common)


# Incremental Current OCV - Sample 2 - Data for 25 deg C
data = data_sam2_25degC
# cycle and step indices to use for SOC
SOC_cycle_indices_discharge = [1,1,2,3,4,5,6,7,8,9,10]
SOC_step_indices_discharge = [4,6,6,6,6,6,6,6,6,6,8]
SOC_cycle_indices_charge = [10,10,11,12,13,14,15,16,17,18,19,19] # could add ten to the beginning
SOC_step_indices_charge = [8,10,10,10,10,10,10,10,10,10,13,16] # could add six to the beginning

#
TestTime = data['Test_Time(s)']
Voltage = data['Voltage(V)']
dV_dt = data['dV/dt(V/s)']
Current = data['Current(A)']
StepIndex = data['Step_Index']
CycleIndex = data['Cycle_Index']
ChargeCapacity = data['Charge_Capacity(Ah)']
DischargeCapacity = data['Discharge_Capacity(Ah)']

# OCV and SOC arrays
size_discharge = len(SOC_cycle_indices_discharge)
discharge_OCV = np.empty(size_discharge)
discharge_SOC = np.empty(size_discharge)
size_charge = len(SOC_cycle_indices_charge)
charge_OCV = np.empty(size_charge)
charge_SOC = np.empty(size_charge)

# Discharge
Qmax = max(DischargeCapacity)
Qmin = min(DischargeCapacity)
if Qnom_flag == 1:
    Qmax = Qnom
    Qmin = 0
for i in range(size_discharge):
    data_to_use = data[(data['Cycle_Index'] == SOC_cycle_indices_discharge[i]) & (data['Step_Index'] == SOC_step_indices_discharge[i])]
    discharge_OCV[i] = data_to_use['Voltage(V)'].iloc[-1]
    discharge_SOC[i] = (Qmax-data_to_use['Discharge_Capacity(Ah)'].iloc[-1])/(Qmax-Qmin)

# Charge
Qmax = max(ChargeCapacity)
Qmin = min(ChargeCapacity)
if Qnom_flag == 1:
    Qmax = Qnom
    Qmin = 0
for i in range(size_charge):
    data_to_use = data[(data['Cycle_Index'] == SOC_cycle_indices_charge[i]) & (data['Step_Index'] == SOC_step_indices_charge[i])]
    charge_OCV[i] = data_to_use['Voltage(V)'].iloc[-1]
    charge_SOC[i] = (data_to_use['Charge_Capacity(Ah)'].iloc[-1]-Qmin)/(Qmax-Qmin)

# Assume: soc1, ocv1 and soc2, ocv2 are your input data
interp1 = interp1d(discharge_SOC, discharge_OCV, kind='cubic', bounds_error=False, fill_value='extrapolate')
interp2 = interp1d(charge_SOC, charge_OCV, kind='cubic', bounds_error=False, fill_value='extrapolate')

ocv1_interp = interp1(SOC_common)
ocv2_interp = interp2(SOC_common)

OCV_sam2_25degC = (ocv1_interp+ocv2_interp)/2
dOCV_dSOC_sam2_25degC = np.gradient(OCV_sam2_25degC, SOC_common)


# Incremental Current OCV - Sample 2 - Data for 45 deg C
data = data_sam2_45degC
# cycle and step indices to use for SOC
SOC_cycle_indices_discharge = [1,1,2,3,4,5,6,7,8,9,10]
SOC_step_indices_discharge = [4,6,6,6,6,6,6,6,6,6,6]
SOC_cycle_indices_charge = [10,11,12,13,14,15,16,17,18,19,19] # could add ten to the beginning
SOC_step_indices_charge = [6,10,10,10,10,10,10,10,10,13,16] # could add six to the beginning

#
TestTime = data['Test_Time(s)']
Voltage = data['Voltage(V)']
dV_dt = data['dV/dt(V/s)']
Current = data['Current(A)']
StepIndex = data['Step_Index']
CycleIndex = data['Cycle_Index']
ChargeCapacity = data['Charge_Capacity(Ah)']
DischargeCapacity = data['Discharge_Capacity(Ah)']

# OCV and SOC arrays
size_discharge = len(SOC_cycle_indices_discharge)
discharge_OCV = np.empty(size_discharge)
discharge_SOC = np.empty(size_discharge)
size_charge = len(SOC_cycle_indices_charge)
charge_OCV = np.empty(size_charge)
charge_SOC = np.empty(size_charge)

# Discharge
Qmax = max(DischargeCapacity)
Qmin = min(DischargeCapacity)
if Qnom_flag == 1:
    Qmax = Qnom
    Qmin = 0
for i in range(size_discharge):
    data_to_use = data[(data['Cycle_Index'] == SOC_cycle_indices_discharge[i]) & (data['Step_Index'] == SOC_step_indices_discharge[i])]
    discharge_OCV[i] = data_to_use['Voltage(V)'].iloc[-1]
    discharge_SOC[i] = (Qmax-data_to_use['Discharge_Capacity(Ah)'].iloc[-1])/(Qmax-Qmin)

# Charge
Qmax = max(ChargeCapacity)
Qmin = min(ChargeCapacity)
if Qnom_flag == 1:
    Qmax = Qnom
    Qmin = 0
for i in range(size_charge):
    data_to_use = data[(data['Cycle_Index'] == SOC_cycle_indices_charge[i]) & (data['Step_Index'] == SOC_step_indices_charge[i])]
    charge_OCV[i] = data_to_use['Voltage(V)'].iloc[-1]
    charge_SOC[i] = (data_to_use['Charge_Capacity(Ah)'].iloc[-1]-Qmin)/(Qmax-Qmin)

# Assume: soc1, ocv1 and soc2, ocv2 are your input data
interp1 = interp1d(discharge_SOC, discharge_OCV, kind='cubic', bounds_error=False, fill_value='extrapolate')
interp2 = interp1d(charge_SOC, charge_OCV, kind='cubic', bounds_error=False, fill_value='extrapolate')

ocv1_interp = interp1(SOC_common)
ocv2_interp = interp2(SOC_common)

OCV_sam2_45degC = (ocv1_interp+ocv2_interp)/2
dOCV_dSOC_sam2_45degC = np.gradient(OCV_sam2_45degC, SOC_common)


#### Exporting and plotting
OCV = (OCV_sam1_0degC + OCV_sam1_25degC + OCV_sam1_45degC + OCV_sam2_0degC) / 4
dOCV_dSOC = np.gradient(OCV, SOC_common)
# export data
np.savez('OCV_dOCV_dSOC_SOC_Incremental', v1=OCV, v2=dOCV_dSOC, v3=SOC_common)
np.savez('OCV_dOCV_dSOC_SOC_Incremental_sam1_0degC', v1=OCV_sam1_0degC, v2=dOCV_dSOC_sam1_0degC, v3=SOC_common)
np.savez('OCV_dOCV_dSOC_SOC_Incremental_sam1_25degC', v1=OCV_sam1_25degC, v2=dOCV_dSOC_sam1_25degC, v3=SOC_common)
np.savez('OCV_dOCV_dSOC_SOC_Incremental_sam1_45degC', v1=OCV_sam1_45degC, v2=dOCV_dSOC_sam1_45degC, v3=SOC_common)
np.savez('OCV_dOCV_dSOC_SOC_Incremental_sam2_0degC', v1=OCV_sam2_0degC, v2=dOCV_dSOC_sam2_0degC, v3=SOC_common)
np.savez('OCV_dOCV_dSOC_SOC_Incremental_sam2_25degC', v1=OCV_sam2_25degC, v2=dOCV_dSOC_sam2_25degC, v3=SOC_common)
np.savez('OCV_dOCV_dSOC_SOC_Incremental_sam2_45degC', v1=OCV_sam2_45degC, v2=dOCV_dSOC_sam2_45degC, v3=SOC_common)

np.savez('OCV_dOCV_dSOC_SOC_Incremental_0degC', v1=1/2*(OCV_sam1_0degC+OCV_sam2_0degC), v2=1/2*(dOCV_dSOC_sam1_0degC+dOCV_dSOC_sam2_0degC), v3=SOC_common)
np.savez('OCV_dOCV_dSOC_SOC_Incremental_25degC', v1=1/2*(OCV_sam1_25degC+OCV_sam2_25degC), v2=1/2*(dOCV_dSOC_sam1_25degC+dOCV_dSOC_sam2_25degC), v3=SOC_common)
np.savez('OCV_dOCV_dSOC_SOC_Incremental_45degC', v1=1/2*(OCV_sam1_45degC+OCV_sam2_45degC), v2=1/2*(dOCV_dSOC_sam1_45degC+dOCV_dSOC_sam2_45degC), v3=SOC_common)

# export data to csv
data_to_export = {
    'SOC': SOC_common,
    'OCV_sam1_0degC' : OCV_sam1_0degC,
    'OCV_sam1_25degC' : OCV_sam1_25degC,
    'OCV_sam1_45degC' : OCV_sam1_45degC,
    'OCV_sam2_0degC' : OCV_sam2_0degC,
    'OCV_sam2_25degC' : OCV_sam2_25degC,
    'OCV_sam2_45degC' : OCV_sam2_45degC
}
df_export = pd.DataFrame(data_to_export)
df_export.to_csv('IncrementalCurrentOCVvsSOC.csv', index=False)


# plotting
fig, axs = plt.subplots(2,1,figsize=(8, 5), sharex=True)
# axs[0].plot(SOC_common,OCV,label='OCV vs SOC',color='blue')
axs[0].plot(SOC_common,OCV_sam1_0degC,label='Sam1 0degC',color='blue')
# Now show all at once
# axs[0].set_xlabel('SOC')
axs[0].set_ylabel('OCV (V)')
axs[0].set_xlim(0,1)
axs[0].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
axs[0].grid(True)
axs[0].set_title(f'Incremental Current OCV and dOCV/dSOC vs SOC')
axs[0].legend()

axs[1].plot(SOC_common,dOCV_dSOC_sam1_0degC,color='blue',label='d(OCV)/d(SOC)')
axs[1].set_xlabel('SOC')
axs[1].set_ylabel('dOCV/dSOC (V)')
axs[1].set_xlim(0,1)
axs[1].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
axs[1].grid(True)

# plotting
fig, axs = plt.subplots(figsize=(8, 5), sharex=True)
axs.plot(SOC_common,OCV_sam1_0degC_discharge,linestyle='--',color='blue',label='Discharge Sam1 0degC')
axs.plot(SOC_common,OCV_sam1_0degC_charge,linestyle='--',color='black',label='Charge Sam1 0degC')
axs.plot(SOC_common,OCV_sam1_0degC,label='Sam1 0degC',color='blue')
# axs.plot(SOC_common,OCV,label='Average OCV',color='blue')

# OCV_sam1_0degC = (ocv1_interp+ocv2_interp)/2
# dOCV_dSOC_sam1_0degC = np.gradient(OCV_sam1_0degC, SOC_common)
# OCV_sam1_0degC_discharge = ocv1_interp
# OCV_sam1_0degC_charge = ocv2_interp

# Now show all at once
plt.xlabel('SOC')
plt.ylabel('OCV (V)')
plt.title('Incremental Current OCV vs SOC')
plt.xlim(0,1)
plt.xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])
plt.legend()

plt.show()