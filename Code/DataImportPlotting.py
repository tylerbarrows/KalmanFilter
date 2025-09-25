##
import pandas as pd 
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy.optimize import curve_fit

# Initial Capacity - Sample 1
# df = pd.read_excel('../Data/10_16_2015_Initial capacity_SP20-1.xlsx', sheet_name='Sheet1')

# Initial Capacity - Sample 2
# df = pd.read_excel('../Data/10_16_2015_Initial capacity_SP20-3.xls', sheet_name='Channel_1-016')

## --------------------------------- ##

# Low current OCV - Sample 1 - Data for 0 deg C
# df1 = pd.read_excel('../Data/02_24_2016_SP20-1_0C_lowcurrentOCV.xls', sheet_name='Channel_1-005_1')
# df2 = pd.read_excel('../Data/02_24_2016_SP20-1_0C_lowcurrentOCV.xls', sheet_name='Channel_1-005_2')
# df = pd.concat([df1, df2], ignore_index=True)

# DOESN'T WORK
# Low current OCV - Sample 1 - Data for 25 deg C
# df1 = pd.read_excel('../Data/11_5_2015_low current OCV test_SP20-1.xls', sheet_name='Channel_1-005_1')
# df2 = pd.read_excel('../Data/11_5_2015_low current OCV test_SP20-1.xls', sheet_name='Channel_1-005_2')
# df = pd.concat([df1, df2], ignore_index=True)

# DOESN'T WORK
# Low current OCV - Sample 1 - Data for 45 deg C 
# df1 = pd.read_excel('../Data/11_21_2015_low current OCV test_SP20-1.xls', sheet_name='Channel_1-005_1')
# df2 = pd.read_excel('../Data/11_21_2015_low current OCV test_SP20-1.xls', sheet_name='Channel_1-005_2')
# df = pd.concat([df1, df2], ignore_index=True)

# Low current OCV - Sample 2 - Data for 0 deg C
# df1 = pd.read_excel('../Data/03_03_2016_SP20-3_0C_lowcurrentOCV.xls', sheet_name='Channel_1-005_1')
# df2 = pd.read_excel('../Data/03_03_2016_SP20-3_0C_lowcurrentOCV.xls', sheet_name='Channel_1-005_2')
# df3 = pd.read_excel('../Data/03_03_2016_SP20-3_0C_lowcurrentOCV.xls', sheet_name='Channel_1-005_3')
# df = pd.concat([df1, df2, df3], ignore_index=True)

# Low current OCV - Sample 2 - Data for 25 deg C 
# df1 = pd.read_excel('../Data/11_16_2015_low current OCV test_SP20-3.xls', sheet_name='Channel_1-004_1')
# df2 = pd.read_excel('../Data/11_16_2015_low current OCV test_SP20-3.xls', sheet_name='Channel_1-004_2')
# df = pd.concat([df1, df2], ignore_index=True)

# DOESN'T WORK
# Low current OCV - Sample 2 - Data for 45 deg C 
# df1 = pd.read_excel('../Data/11_21_2015_low current OCV test_SP20-3_samallcurrent.xls', sheet_name='Channel_1-004_1')
# df2 = pd.read_excel('../Data/11_21_2015_low current OCV test_SP20-3_samallcurrent.xls', sheet_name='Channel_1-004_2')
# df = pd.concat([df1, df2], ignore_index=True)

## --------------------------------- ##

# Incremental Current OCV - Sample 1 - Data for 0 deg C
# df1 = pd.read_excel('../Data/02_26_2016_SP20-1_0C_incrementalOCV.xls',sheet_name='Channel_1-005_1')
# df2 = pd.read_excel('../Data/02_26_2016_SP20-1_0C_incrementalOCV.xls',sheet_name='Channel_1-005_2')
# df3 = pd.read_excel('../Data/02_26_2016_SP20-1_0C_incrementalOCV.xls',sheet_name='Channel_1-005_3')
# df = pd.concat([df1, df2, df3], ignore_index=True)

# Incremental Current OCV - Sample 1 - Data for 25 deg C
# df1 = pd.read_excel('../Data/12_2_2015_Incremental OCV test_SP20-1.xlsx',sheet_name='Channel_1-005_1')
# df2 = pd.read_excel('../Data/12_2_2015_Incremental OCV test_SP20-1.xlsx',sheet_name='Channel_1-005_2')
# df3 = pd.read_excel('../Data/12_2_2015_Incremental OCV test_SP20-1.xlsx',sheet_name='Channel_1-005_3')
# df = pd.concat([df1, df2, df3], ignore_index=True)

# Incremental Current OCV - Sample 1 - Data for 45 deg C
# df1 = pd.read_excel('../Data/12_09_2015_Incremental OCV test_SP20-1.xlsx',sheet_name='Channel_1-005_1')
# df2 = pd.read_excel('../Data/12_09_2015_Incremental OCV test_SP20-1.xlsx',sheet_name='Channel_1-005_2')
# df3 = pd.read_excel('../Data/12_09_2015_Incremental OCV test_SP20-1.xlsx',sheet_name='Channel_1-005_3')
# df = pd.concat([df1, df2, df3], ignore_index=True)

# Incremental Current OCV - Sample 2 - Data for 0 deg C
# df1 = pd.read_excel('../Data/03_09_2016_SP20-3_0C_incrementalOCV.xls',sheet_name='Channel_1-005_1')
# df2 = pd.read_excel('../Data/03_09_2016_SP20-3_0C_incrementalOCV.xls',sheet_name='Channel_1-005_2')
# df3 = pd.read_excel('../Data/03_09_2016_SP20-3_0C_incrementalOCV.xls',sheet_name='Channel_1-005_3')
# df = pd.concat([df1, df2, df3], ignore_index=True)

# Incremental Current OCV - Sample 2 - Data for 25 deg C
# df1 = pd.read_excel('../Data/12_2_2015_Incremental OCV test_SP20-3.xlsx',sheet_name='Channel_1-006_1')
# df2 = pd.read_excel('../Data/12_2_2015_Incremental OCV test_SP20-3.xlsx',sheet_name='Channel_1-006_2')
# df3 = pd.read_excel('../Data/12_2_2015_Incremental OCV test_SP20-3.xlsx',sheet_name='Channel_1-006_3')
# df = pd.concat([df1, df2, df3], ignore_index=True)

# Incremental Current OCV - Sample 2 - Data for 45 deg C
# df1 = pd.read_excel('../Data/12_09_2015_Incremental OCV test_SP20-3.xlsx',sheet_name='Channel_1-006_1')
# df2 = pd.read_excel('../Data/12_09_2015_Incremental OCV test_SP20-3.xlsx',sheet_name='Channel_1-006_2')
# df3 = pd.read_excel('../Data/12_09_2015_Incremental OCV test_SP20-3.xlsx',sheet_name='Channel_1-006_3')
# df = pd.concat([df1, df2, df3], ignore_index=True)

## --------------------------------- ##

# DST - Data for 0 deg C (50 % SOC)
# df = pd.read_excel('../Data/DST/SP2_0C_DST/02_24_2016_SP20-2_0C_DST_50SOC.xls',sheet_name='Channel_1-006')

# DST - Data for 0 deg C (80 % SOC)
df = pd.read_excel('../Data/DST/SP2_0C_DST/02_24_2016_SP20-2_0C_DST_80SOC.xls',sheet_name='Channel_1-006')

## --------------------------------- ##
# FUDS - Data for 0 deg C (50 % SOC)
# df = pd.read_excel('../Data/FUDS/SP2_0C_FUDS/02_25_2016_SP20-2_0C_FUDS_50SOC.xls',sheet_name='Channel_1-006')

# finding number of tests
num_tests = df['Step_Index'].max()
# num_tests = df['Cycle_Index'].max()
print(f"Number of tests: {num_tests}")

# Step 3: Group data by test index
# grouped = df.groupby('Step_Index')
grouped = df.groupby('Cycle_Index')

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


##
# figs = []

# for test_idx, test_data in grouped:
#     formatter = ScalarFormatter(useMathText=True)
#     # formatter.set_scientific(True)
#     # formatter.set_powerlimits((-np.inf, np.inf))  # Always use scientific notation
#     formatter.set_useOffset(False)  # Turn off the offset text (like 1e-5 at top)

#     fig, axs = plt.subplots(3, 1, figsize=(8, 5), sharex=True)

#     axs[0].plot(test_data['Step_Time(s)'], test_data['Voltage(V)'], marker='o')
#     axs[0].set_title(f'Test {test_idx}')
#     axs[0].set_xlabel('Time')
#     axs[0].set_ylabel('Voltage (V)')
#     axs[0].grid(True)
#     # axs[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
#     axs[0].yaxis.set_major_formatter(formatter)

#     axs[1].plot(test_data['Step_Time(s)'], test_data['dV/dt(V/s)'], marker='o')
#     axs[1].set_xlabel('Time')
#     axs[1].set_ylabel('dV/dt (V/s)')
#     axs[1].grid(True)
#     axs[1].yaxis.set_major_formatter(formatter)

#     axs[2].plot(test_data['Step_Time(s)'], test_data['Current(A)'], marker='o')
#     axs[2].set_xlabel('Time')
#     axs[2].set_ylabel('Current (A)')
#     axs[2].grid(True)
#     axs[2].yaxis.set_major_formatter(formatter)

#     figs.append(fig)  # Keep reference to prevent garbage collection

## only voltage, current, and dv/dt
fig, axs = plt.subplots(3, 1, figsize=(8, 5), sharex=True)
size_for_markers_plot = 2
size_for_markers_scatter = 22

axs[0].plot(df['Test_Time(s)'], df['Voltage(V)'], marker='o', markersize=size_for_markers_plot)
# axs[0].set_title(f'Low Current Test')
# axs[0].set_title(f'Incremental OCV Test')
axs[0].set_title(f'DST Test')
# axs[0].set_xlabel('Time')
axs[0].set_ylabel('Voltage (V)')
axs[0].grid(True)

axs[2].plot(df['Test_Time(s)'], df['dV/dt(V/s)'], marker='o', markersize=size_for_markers_plot)
axs[2].set_xlabel('Time (s)')
axs[2].set_ylabel('dV/dt (V/s)')
axs[2].grid(True)

axs[1].plot(df['Test_Time(s)'], -df['Current(A)'], marker='o', markersize=size_for_markers_plot)
# axs[1].set_xlabel('Time')
axs[1].set_ylabel('Current (A)')
axs[1].grid(True)

plt.show()