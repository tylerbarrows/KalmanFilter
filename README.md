# KalmanFilter
Kalman filter applied to a lithium ion battery state of charge application. Code section includes all code used from building the model (including finding the OCV-SOC relationship to finding equivalent circuit parameters) to simulating online SOC estimation. 

# Format
The respository should be downloaded and unzipped. Both /Code/ and /Data/ should be located in the same directory at the same level:
model/
├── code/
└── data/

The code will output .csv and .npz files in the same directory (e.g., /Code/). Ideally this is set up differently where the files produced by the code are stored somewhere else. Even more ideally, code related to the model should be located seperately from the SOC online simulation code. All of the files produced by the code (including OCV-SOC relationships and parameters) are present which represent the latest runs I ran from my computer. 

# Data
Battery data was used from the University of Maryland (https://calce.umd.edu/battery-data). The data was for the INR 18650-20R Battery cell (first one on the page). The Low Current and Incremental Current data belongs in /Data/. Some of the data is there but not all. The DST and FUDS data is in /Data/DST/ and /Data/FUDS/ respectively and all of that data is there. 
