# Activity Recognition Model

This project is part of a research thesis focused on developing a model for activity recognition based on users' biomechanical data.

## Project Structure
- `data_logs/raw` - Users motion logs for trainining 
- `data_logs/raw/test_data` - Users motion logs for validation 
- `data_logs/processed` - Clustering Results 
- `data_logs/results/recognition_results` - Results of the validation process
- `model_project/src/` - Model code
- `model_project/src/data_processing/` - Data processing codes
- `model_project/src/model.ipynb` - Jupyter notebook to run the model 

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## Usage

1. Clone the repository:
    ```
    git clone https://github.com/VSecLab/act_recon.git
    ```
2. Install dependencies:
    ```
    pip install -r requirements.txt
    ```
3. Open "model_project/src/model.ipynb" and run the model step by step.

---

The VR_grim_build_final.apk file included in this repository is a Unity-based VR game designed for standalone VR headsets (e.g., Meta Quest). It serves as the interactive component of the project and is used for gameplay and data collection purposes.

### How to Use

1. Install the APK on your VR headset using ADB or a sideloading tool such as SideQuest (suggested).

2. Launch the game from your headset's apps menu once installation is complete.

Make sure your headset allows installations from unknown sources (check your device settings).  

### Data Collection

If data logging is enabled, gameplay data will be automatically saved on the device.
You can retrieve the captured data from the following path on the headset: 
Android/data/com.YourCompanyName.YourGameName/files/

    Android/data/com.UnityTechnologies.com.unity.template.urpblank/files/Logs


## Contact

For questions or collaboration, please contact [fr.grimaldi@outlook.com].
