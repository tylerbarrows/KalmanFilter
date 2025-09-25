## Comparing OCV vs SOC from Low Current and Incremental Current methods
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

# First load low current method
SOC_OCV_LowCurrent = np.load('OCV_dOCV_dSOC_SOC_LowCurrent.npz')
SOC_OCV_LowCurrent_sam1_0degC = np.load('OCV_dOCV_dSOC_SOC_LowCurrent_sam1_0degC.npz')
SOC_OCV_LowCurrent_sam2_0degC = np.load('OCV_dOCV_dSOC_SOC_LowCurrent_sam2_0degC.npz')
SOC_OCV_LowCurrent_sam2_25degC = np.load('OCV_dOCV_dSOC_SOC_LowCurrent_sam2_25degC.npz')

OCV_LowCurrent = SOC_OCV_LowCurrent['v1']
dOCV_dSOC_LowCurrent = SOC_OCV_LowCurrent['v2']
# SOC_LowCurrent = SOC_OCV_LowCurrent['v3']
SOC = SOC_OCV_LowCurrent['v3']

OCV_LowCurrent_sam1_0degC = SOC_OCV_LowCurrent_sam1_0degC['v1']
dOCV_dSOC_LowCurrent_sam1_0degC = SOC_OCV_LowCurrent_sam1_0degC['v2']
# SOC_LowCurrent_sam1_0degC = SOC_OCV_LowCurrent_sam1_0degC['v3']

OCV_LowCurrent_sam2_0degC = SOC_OCV_LowCurrent_sam2_0degC['v1']
dOCV_dSOC_LowCurrent_sam2_0degC = SOC_OCV_LowCurrent_sam2_0degC['v2']
# SOC_LowCurrent_sam2_0degC = SOC_OCV_LowCurrent_sam2_0degC['v3']

OCV_LowCurrent_sam2_25degC = SOC_OCV_LowCurrent_sam2_25degC['v1']
dOCV_dSOC_LowCurrent_sam2_25degC = SOC_OCV_LowCurrent_sam2_25degC['v2']
# SOC_LowCurrent_sam2_25degC = SOC_OCV_LowCurrent_ssam2_25degC['v3']

# Second load incremental current method
SOC_OCV_Incremental = np.load('OCV_dOCV_dSOC_SOC_Incremental.npz')
SOC_OCV_Incremental_sam1_0degC = np.load('OCV_dOCV_dSOC_SOC_Incremental_sam1_0degC.npz')
SOC_OCV_Incremental_sam1_25degC = np.load('OCV_dOCV_dSOC_SOC_Incremental_sam1_25degC.npz')
SOC_OCV_Incremental_sam1_45degC = np.load('OCV_dOCV_dSOC_SOC_Incremental_sam1_45degC.npz')
SOC_OCV_Incremental_sam2_0degC = np.load('OCV_dOCV_dSOC_SOC_Incremental_sam2_0degC.npz')
SOC_OCV_Incremental_sam2_25degC = np.load('OCV_dOCV_dSOC_SOC_Incremental_sam2_25degC.npz')
SOC_OCV_Incremental_sam2_45degC = np.load('OCV_dOCV_dSOC_SOC_Incremental_sam2_45degC.npz')

# Finish this - basically want to compare methods
# especially 0.1 < SOC < 0.9
# then want to expand RC components
# then can summarize all this work and then get started on Kalman filter

OCV_IncrementalCurrent = SOC_OCV_Incremental['v1']
dOCV_dSOC_IncrementalCurrent = SOC_OCV_Incremental['v2']
# SOC_IncrementalCurrent = SOC_OCV_Incremental['v3']

OCV_IncrementalCurrent_sam1_0degC = SOC_OCV_Incremental_sam1_0degC['v1']
dOCV_dSOC_IncrementalCurrent_sam1_0degC = SOC_OCV_Incremental_sam1_0degC['v2']
# SOC_IncrementalCurrent_sam1_0degC = SOC_OCV_Incremental_sam1_0degC['v3']

OCV_IncrementalCurrent_sam1_25degC = SOC_OCV_Incremental_sam1_25degC['v1']
dOCV_dSOC_IncrementalCurrent_sam1_25degC = SOC_OCV_Incremental_sam1_25degC['v2']
# SOC_IncrementalCurrent_sam1_25degC = SOC_OCV_Incremental_sam1_25degC['v3']

OCV_IncrementalCurrent_sam1_45degC = SOC_OCV_Incremental_sam1_45degC['v1']
dOCV_dSOC_IncrementalCurrent_sam1_45degC = SOC_OCV_Incremental_sam1_45degC['v2']
# SOC_IncrementalCurrent_sam1_45degC = SOC_OCV_Incremental_sam1_45degC['v3']

OCV_IncrementalCurrent_sam2_0degC = SOC_OCV_Incremental_sam2_0degC['v1']
dOCV_dSOC_IncrementalCurrent_sam2_0degC = SOC_OCV_Incremental_sam2_0degC['v2']
# SOC_IncrementalCurrent_sam2_0degC = SOC_OCV_Incremental_sam2_0degC['v3']

OCV_IncrementalCurrent_sam2_25degC = SOC_OCV_Incremental_sam2_25degC['v1']
dOCV_dSOC_IncrementalCurrent_sam2_25degC = SOC_OCV_Incremental_sam2_25degC['v2']
# SOC_IncrementalCurrent_sam2_25degC = SOC_OCV_Incremental_sam2_25degC['v3']

OCV_IncrementalCurrent_sam2_45degC = SOC_OCV_Incremental_sam2_45degC['v1']
dOCV_dSOC_IncrementalCurrent_sam2_45degC = SOC_OCV_Incremental_sam2_45degC['v2']
# SOC_IncrementalCurrent_sam2_45degC = SOC_OCV_Incremental_sam2_45degC['v3']

# Indices
SampleIndices = [1,1,1,2,2,2]
TempIndices = [0,25,45,0,24,45]
LowCurrentIndices = [1,0,0,1,1,0]
IncrementalCurrentIndices = [1,1,1,1,1,1]

# trying a class approach for this
class Measurement:
    def __init__(self, samNumber, temperature, OCV, dOCV_dSOC):
        self.samNumber = samNumber
        self.temperature = temperature
        self.OCV = OCV
        self.dOCV_dSOC = dOCV_dSOC

class Sample:
    def __init__(self, name):
        self.name = name
        self.measurements = []  # List of Measurement objects

    def add_measurement(self, samNumber, temperature, OCV, dOCV_dSOC):
        self.measurements.append(Measurement(samNumber, temperature, OCV, dOCV_dSOC))

# Container for all samples
class ExperimentData:
    def __init__(self):
        self.samples = {}  # key = sample name

    def add_sample(self, sample):
        self.samples[sample.name] = sample

data = ExperimentData()

# Sample A with two temperature measurements
sampleA = Sample('LowCurrent')
sampleA.add_measurement(1, 0, OCV=OCV_LowCurrent_sam1_0degC, dOCV_dSOC=dOCV_dSOC_LowCurrent_sam1_0degC)
sampleA.add_measurement(2, 0, OCV=OCV_LowCurrent_sam2_0degC, dOCV_dSOC=dOCV_dSOC_LowCurrent_sam2_0degC)
sampleA.add_measurement(2, 25, OCV=OCV_LowCurrent_sam2_25degC, dOCV_dSOC=dOCV_dSOC_LowCurrent_sam2_25degC)
data.add_sample(sampleA)

print(data.samples['LowCurrent'].measurements[0].samNumber)

# Sample B
sampleB = Sample('IncrementalCurrent')
sampleB.add_measurement(1, 0, OCV=OCV_IncrementalCurrent_sam1_0degC, dOCV_dSOC=dOCV_dSOC_IncrementalCurrent_sam1_0degC)
sampleB.add_measurement(1, 25, OCV=OCV_IncrementalCurrent_sam1_25degC, dOCV_dSOC=dOCV_dSOC_IncrementalCurrent_sam1_25degC)
sampleB.add_measurement(1, 45, OCV=OCV_IncrementalCurrent_sam1_45degC, dOCV_dSOC=dOCV_dSOC_IncrementalCurrent_sam1_45degC)
sampleB.add_measurement(2, 0, OCV=OCV_IncrementalCurrent_sam2_0degC, dOCV_dSOC=dOCV_dSOC_IncrementalCurrent_sam2_0degC)
sampleB.add_measurement(2, 25, OCV=OCV_IncrementalCurrent_sam2_25degC, dOCV_dSOC=dOCV_dSOC_IncrementalCurrent_sam2_25degC)
sampleB.add_measurement(2, 45, OCV=OCV_IncrementalCurrent_sam2_45degC, dOCV_dSOC=dOCV_dSOC_IncrementalCurrent_sam2_45degC)
data.add_sample(sampleB)


# trim SOC indices if we want to - comment out if don't want to
SOC_trim_indices = np.where((SOC >= 0.1) & (SOC <= 0.9))[0]  # Indices to keep
# SOC_trim_indices = np.where((SOC >= 0.1) & (SOC <= 1))[0]  # Indices to keep
for sample in data.samples.values():
    for measurement in sample.measurements:
        measurement.OCV = np.array(measurement.OCV)[SOC_trim_indices]
        measurement.dOCV_dSOC = np.array(measurement.dOCV_dSOC)[SOC_trim_indices]

SOC = SOC[SOC_trim_indices]
OCV_LowCurrent = OCV_LowCurrent[SOC_trim_indices]
OCV_IncrementalCurrent = OCV_IncrementalCurrent[SOC_trim_indices]
dOCV_dSOC_LowCurrent = dOCV_dSOC_LowCurrent[SOC_trim_indices]
dOCV_dSOC_IncrementalCurrent = dOCV_dSOC_IncrementalCurrent[SOC_trim_indices]

# Average comparison
fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
size_for_markers_plot = 2
size_for_markers_scatter = 22

axs[0].plot(SOC, OCV_LowCurrent, label='Low Current')
axs[0].plot(SOC, OCV_IncrementalCurrent, label='Incremental Current')
axs[0].set_title(f'Low Current vs Incremental Current')
axs[0].set_xlabel('SOC')
axs[0].set_ylabel('OCV (V)')
axs[0].grid(True)
axs[0].legend()
axs[0].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

axs[1].plot(SOC, dOCV_dSOC_LowCurrent, label='Low Current')
axs[1].plot(SOC, dOCV_dSOC_IncrementalCurrent, label='Incremental Current')
axs[1].set_title(f'Low Current vs Incremental Current')
axs[1].set_xlabel('SOC')
axs[1].set_ylabel('dOCV/dSOC (V)')
axs[1].grid(True)
# axs[1].legend()
axs[1].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

# # Average comparison
# fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
# size_for_markers_plot = 2
# size_for_markers_scatter = 22

# axs[0].plot(SOC, OCV_LowCurrent, label='Low Current')
# axs[0].plot(SOC, OCV_IncrementalCurrent, label='Incremental Current')
# axs[0].set_title(f'Low Current vs Incremental Current (Trimmed)')
# axs[0].set_xlabel('SOC')
# axs[0].set_ylabel('OCV (V)')
# axs[0].grid(True)
# axs[0].legend()
# axs[0].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

# axs[1].plot(SOC, dOCV_dSOC_LowCurrent, label='Low Current')
# axs[1].plot(SOC, dOCV_dSOC_IncrementalCurrent, label='Incremental Current')
# axs[1].set_title(f'Low Current vs Incremental Current')
# axs[1].set_xlabel('SOC')
# axs[1].set_ylabel('dOCV/dSOC (V)')
# axs[1].grid(True)
# # axs[1].legend()
# axs[1].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])


def plot_ocv_and_derivative(data, samples=None, samNumbers=None, temperatures=None, plotTitle=None):
    """
    Plot voltage vs SOC for given sample names and/or temperatures.
    If `samples` or `temperatures` is None, include all.
    """
    fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)

    for sample_name, sample in data.samples.items():
        if samples and sample_name not in samples:
            continue
        for n in sample.measurements:
            if samNumbers and n.samNumber not in samNumbers:
                continue
            if temperatures and n.temperature not in temperatures:
                continue
            label = f'{sample_name} - #{n.samNumber} - {n.temperature}°C'
            # plt.plot(m.soc, m.voltage, label=label)
            axs[0].plot(SOC, n.OCV, label=label)
            axs[1].plot(SOC, n.dOCV_dSOC, label=label)

    # plt.xlabel("State of Charge (%)")
    # plt.ylabel("Voltage (V)")
    # plt.title("Voltage vs SOC")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    
    # axs[0].plot(SOC, OCV_IncrementalCurrent, label='Incremental Current')
    axs[0].set_title(plotTitle)
    axs[0].set_xlabel('SOC')
    axs[0].set_ylabel('OCV (V)')
    axs[0].grid(True)
    axs[0].legend()
    axs[0].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

    # axs[1].plot(SOC, dOCV_dSOC_LowCurrent, label='Low Current')
    # axs[1].plot(SOC, dOCV_dSOC_IncrementalCurrent, label='Incremental Current')
    # axs[1].set_title(f'Low Current vs Incremental Current')
    axs[1].set_xlabel('SOC')
    axs[1].set_ylabel('dOCV/dSOC (V)')
    axs[1].grid(True)
    # axs[1].legend()
    axs[1].set_xticks([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])

    # plt.show()

# for sample_name, sample in data.samples.items():
#     print(sample_name)
#     print(sample.measurements[0].temperature)

plot_ocv_and_derivative(data, samNumbers=[1], temperatures=[0], plotTitle='Sample 1, 0 deg. C')
plot_ocv_and_derivative(data, samNumbers=[2], temperatures=[0], plotTitle='Sample 2, 0 deg. C')
plot_ocv_and_derivative(data, samNumbers=[2], temperatures=[25], plotTitle='Sample 2, 25 deg. C')
plot_ocv_and_derivative(data, samNumbers=[1], temperatures=[0,25,45], plotTitle='Sample 1')
plot_ocv_and_derivative(data, samNumbers=[2], temperatures=[0,25,45], plotTitle='Sample 2')
plot_ocv_and_derivative(data, samples='IncrementalCurrent', samNumbers=[1], temperatures=[0,25,45], plotTitle='Incremental Current - Sample 1')
plot_ocv_and_derivative(data, samples='IncrementalCurrent', samNumbers=[2], temperatures=[0,25,45], plotTitle='Incremental Current - Sample 2')
plot_ocv_and_derivative(data, samples='IncrementalCurrent', samNumbers=[1,2], temperatures=[0,25,45], plotTitle='Incremental Current')
plot_ocv_and_derivative(data, samples='LowCurrent', samNumbers=[1,2], temperatures=[0,25,45], plotTitle='Low Current')
plot_ocv_and_derivative(data, samNumbers=[1,2], temperatures=[0,25,45], plotTitle='All')
plot_ocv_and_derivative(data, samNumbers=[1,2], temperatures=[0], plotTitle='0 deg. C')
plot_ocv_and_derivative(data, samNumbers=[1,2], temperatures=[25], plotTitle='25 deg. C')
plot_ocv_and_derivative(data, samNumbers=[1,2], temperatures=[45], plotTitle='45 deg. C')
plot_ocv_and_derivative(data, samples='IncrementalCurrent', samNumbers=[1,2], temperatures=[0], plotTitle='Incremental Current - 0 deg. C')
plot_ocv_and_derivative(data, samples='IncrementalCurrent', samNumbers=[1,2], temperatures=[25], plotTitle='Incremental Current - 25 deg. C')
plot_ocv_and_derivative(data, samples='IncrementalCurrent', samNumbers=[1,2], temperatures=[45], plotTitle='Incremental Current - 45 deg. C')

plt.show()