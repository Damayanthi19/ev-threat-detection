# EV Threat Detection 🚗⚡🔒

A machine learning–powered system to detect potential cyber-attacks on Electric Vehicle (EV) charging stations from incoming communication logs — now with **custom sound alerts** for each type of attack.

---

## 📌 Project Overview

Electric Vehicle charging stations are increasingly network-connected, making them susceptible to threats such as **Man-in-the-Middle (MitM)**, **Spoofing**, and **DoS** attacks.  
This project uses a trained machine learning model to classify logs as **Normal** or **Attack** in real time.

The included `detect.py` script:
- Reads incoming CSV logs (no label required)
- Extracts and prepares features
- Classifies each row
- **Plays unique audio alerts** for each attack type
- Shows probability scores for transparency

---

## 🛠 Tech Stack

- **Language**: Python 3.9+
- **Data Processing**: Pandas, NumPy
- **Modeling**: scikit-learn (Gradient Boosting, pipelines, preprocessing)
- **Serialization**: Joblib (for preprocessor, model, and label encoder)
- **CLI Output Styling**: termcolor, colorama
- **Sound Playback**: playsound / winsound
- **Environment**: Conda / Virtualenv

---

## 📂 Project Structure

