##
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy.optimize import curve_fit

# Data import
df1 = pd.read_excel('../Data/02_24_2016_SP20-1_0C_lowcurrentOCV.xls', sheet_name='Channel_1-005_1')
df2 = pd.read_excel('../Data/02_24_2016_SP20-1_0C_lowcurrentOCV.xls', sheet_name='Channel_1-005_2')
df = pd.concat([df1, df2], ignore_index=True)

# finding number of tests
num_tests = df['Step_Index'].max()
print(f"Number of tests: {num_tests}")

# Step 3: Group data by test index
grouped = df.groupby('Step_Index')

##
figs = []

for test_idx, test_data in grouped:
    formatter = ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-np.inf, np.inf))  # Always use scientific notation
    formatter.set_useOffset(False)  # Turn off the offset text (like 1e-5 at top)

    fig, axs = plt.subplots(3, 1, figsize=(8, 5), sharex=True)

    axs[0].plot(test_data['Step_Time(s)'], test_data['Voltage(V)'], marker='o')
    axs[0].set_title(f'Test {test_idx}')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Voltage (V)')
    axs[0].grid(True)
    # axs[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    axs[0].yaxis.set_major_formatter(formatter)

    axs[1].plot(test_data['Step_Time(s)'], test_data['dV/dt(V/s)'], marker='o')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('dV/dt (V/s)')
    axs[1].grid(True)
    axs[1].yaxis.set_major_formatter(formatter)

    axs[2].plot(test_data['Step_Time(s)'], test_data['Current(A)'], marker='o')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('Current (A)')
    axs[2].grid(True)
    axs[2].yaxis.set_major_formatter(formatter)

    figs.append(fig)  # Keep reference to prevent garbage collection

## Fitting data
data_to_fit = df[df['Step_Index'] == 7]
y_data_to_fit = data_to_fit['Voltage(V)']
x_data_to_fit = data_to_fit['Step_Time(s)']
size = len(y_data_to_fit)
Vo = df[df['Step_Index'] == 6]['Voltage(V)'].iloc[-1]
print('Voltage at end of previous step index: ',Vo)
print('Voltage at beginning of first step index: ', y_data_to_fit.iloc[0])

def double_exponential(t, a1, a2, tau1, tau2):
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)

##
V_threshold_range = np.arange(0.62,0.1,-0.005)
R_track = np.empty(len(V_threshold_range))
a1_track = np.empty(len(V_threshold_range))
a2_track = np.empty(len(V_threshold_range))
tau1_track = np.empty(len(V_threshold_range))
tau2_track = np.empty(len(V_threshold_range))

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
    # print(params)
    # print(covariance)
    # print(tau1)
    # print(tau2)
    ss_res = np.sum((y_data_to_fit_prime - y_fit)**2)
    ss_tot = np.sum((y_data_to_fit_prime - np.mean(y_data_to_fit_prime))**2)
    r_squared = 1 - (ss_res / ss_tot)

    R_track[ind] = r_squared
    a1_track[ind] = a1
    a2_track[ind] = a2
    tau1_track[ind] = tau1
    tau2_track[ind] = tau2

    if r_squared > R2_ticker:
        V_threshold_use = V_threshold
        R2_ticker = r_squared
        guess_to_use = [a1, a2, tau1, tau2]

    ind = ind  + 1

print('Use this as V_threshold: ', V_threshold_use)
print('max R^2: ', R2_ticker)
## Parametric plots
# R^2
fig, axs = plt.subplots(figsize=(8, 5), sharex=True)
axs.scatter(V_threshold_range,R_track,label='R^2')

# Now show all at once
plt.xlabel('V threshold (V)')
plt.ylabel('R^2')

# a1, a2, tau1, tau2
fig, axs = plt.subplots(2,2,figsize=(8, 5), sharex=True)

axs[0,0].scatter(V_threshold_range,a1_track,label='a1')
axs[0,0].set_xlabel('V threshold (V)')
axs[0,0].set_ylabel('a1')

axs[0,1].scatter(V_threshold_range,a2_track,label='a1')
axs[0,1].set_xlabel('V threshold (V)')
axs[0,1].set_ylabel('a2')

axs[1,0].scatter(V_threshold_range,tau1_track,label='a1')
axs[1,0].set_xlabel('V threshold (V)')
axs[1,0].set_ylabel('tau1')

axs[1,1].scatter(V_threshold_range,tau2_track,label='a1')
axs[1,1].set_xlabel('V threshold (V)')
axs[1,1].set_ylabel('tau2')

## Using best fit we found earlier
y_data_to_fit_prime = y_data_to_fit
x_data_to_fit_prime = x_data_to_fit

y_data_to_fit_prime = (y_data_to_fit-y_data_to_fit.iloc[size-1])*-1
V_threshold = V_threshold_use
start_index = y_data_to_fit_prime[y_data_to_fit_prime < V_threshold].index.min()
x_data_to_fit_prime = x_data_to_fit_prime.loc[start_index:].reset_index(drop=True)
y_data_to_fit_prime = y_data_to_fit_prime.loc[start_index:].reset_index(drop=True)
x_data_to_fit_prime = x_data_to_fit_prime - x_data_to_fit_prime.iloc[0]

initial_guess = [0.25,0.1,225,2100]
initial_guess = guess_to_use
params, covariance = curve_fit(double_exponential, x_data_to_fit_prime, y_data_to_fit_prime, p0=initial_guess)
a1, a2, tau1, tau2 = params
y_fit = double_exponential(x_data_to_fit_prime,*params)
Ro = (y_data_to_fit.iloc[0] - V_threshold_use)/0.1
print('Ro: ', Ro)
print('a1: ', a1)
print('a2: ', a2)
print('tau1: ', tau1)
print('tau2: ', tau2)
ss_res = np.sum((y_data_to_fit_prime - y_fit)**2)
ss_tot = np.sum((y_data_to_fit_prime - np.mean(y_data_to_fit_prime))**2)
r_squared = 1 - (ss_res / ss_tot)
print('R^2: ', r_squared)

fig, axs = plt.subplots(figsize=(8, 5), sharex=True)
axs.plot(x_data_to_fit_prime,y_data_to_fit_prime,label='Data')
axs.plot(x_data_to_fit_prime,y_fit,label='Fit')

# Now show all at once
plt.xlabel('Time (s)')
plt.ylabel('Voltage(V)')
plt.legend()

# original data
fig, axs = plt.subplots(figsize=(8, 5), sharex=True)
axs.plot(x_data_to_fit,(y_data_to_fit-y_data_to_fit.iloc[size-1])*-1,label='Data')

# Now show all at once
plt.xlabel('Time (s)')
plt.ylabel('Voltage(V)')
plt.legend()

##
plt.show()