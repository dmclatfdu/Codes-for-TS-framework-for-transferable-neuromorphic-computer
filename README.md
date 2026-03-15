# Transferable Neuromorphic Computer



## 📁 File Structure
There are nine code files and a data folder in total, which are:
### Program files
| File Name | Description |
|-----------|-------------|
| `base_library.py` | Basic functions used across all program files |
| `sim_RC_library.py` | Simulated TiOx-based RC **initial settings & pre-processing** |
| `device_characteristics.py` | Experimental device **characterizations** & simulated reproduction |
| `RC_MG.py` | **Mackey-Glass** one-step prediction tasks |
| `RC_Lorenz.py` | **Lorenz system** recurrent prediction |
| `RC_Arrhythmia.py` | **Arrhythmia** detection (demonstrated on the ECG heartbeat dataset below), modified from Codes in NE2022 of https://github.com/Tsinghua-LEMON-Lab/Reservoir-computing|
| `RC_Voice_Exp_TiOx.py` |  **Experimental spoken digit classification using the TiOx memristor-based RC**  |
| `RC_Voice_Exp_NbOx.py` |  **Experimental spoken digit classification using the NbOx memristor-based RC**|
| `Voice_Inputs.py` | The **supporting files** for input signal generation for **experimental RC** in the spoken digit classification|


### Data file folder
The data file folder stores the **experimentally measured data**. It will also stores the results of RC when running the above programs. We are willing to provide all experimental data to the reviewers and editors, and will make them publicly available upon manuscript publication. In the following describes the basic information of the data (file/folder name shown in the detailed directory).

| File/Folder Name | Description |
|-----------|-------------|
| `Data/Arrhythmia/ECGdataset.mat` | Processed **ECG heartbeat** records dataset (copied from https://github.com/Tsinghua-LEMON-Lab/Reservoir-computing)|
| `Data/Characterization` | The folder for the **characterization (IV/Pulse/Decay) results** of the TiOx and NbOx devices |
| `Data/MG/Exp/TiOx` | The experimental data for the **MG task with TiOx-based RC**, including the generated voltage signal files on the Keysight B1500A and the measured responses. |
| `Data/Voice` | The experimental data for the **spoken digit classification with TiOx/NbOx-based RC**, including the generated voltage signal files on the Keysight B1500A and the measured responses. |


## ⚠️ Notice
**I.** To run the programs successfully, **the following libraries are required**: **SciPy** (1.7.1), **tqdm** (4.62.3), **NumPy** (1.22.4), **Pandas** (1.3.4), **Seaborn** (0.12.2), **Matplotlib** (3.4.3), **Scikit-Learn** (1.0.1), **h5py** (3.7.0), **librosa** (0.10.0). Python version is 3.8.20.

**II.** **Run device_characteristics.py before running RC_MG.py**, since its results are needed for arranging the order of devices.

**III.** The code in **RC_Arrhythmia.py would take a lot of time**, about 3-4 days (we use the AMD Ryzen 5800H and the Intel Core i5-14600KF).

**IV.** The **librosa library** (used in Voice_Inputs.py) **sometimes meets the problem**: osError cannot load library 'libsndfile.dll':error 0x7e. **To solve this problem**, you may have to manually do the following steps: **(1)** **locate the directory which reports the error** (when using anaconda to create an environment, it is most likely .conda/envs/your_env_name/Lib/site-packages); **(2)** create a folder named **_soundfile_data** in the directory; **(3)** put the file **libsndfile_64bit.dll (provided in this repository)** in the _soundfile_data folder. After the procedures above, run the code again to check if the problem is fixed.



















