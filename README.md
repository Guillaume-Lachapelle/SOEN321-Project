# NSL-KDD Attacks Project

This project explores and implements four distinct attacks on the NSL-KDD dataset, which is  used in the domains of **Network Security**, **Information Security**, and **Cyber Security**. The goal is to analyze the impact and effectiveness of each attack on machine learning models trained with this dataset.

## Overview of Performed Attacks

1. **Denial of Service (DoS) Attack**  
   Simulates an overload on the system by overwhelming resources, aiming to disrupt the normal functionality of the network.

2. **Adversarial Attack**  
   Manipulates training data to mislead the model into making incorrect predictions.

3. **Membership Inference Attack**  
   Exploits the model to infer whether a specific data point was part of its training dataset, potentially leaking sensitive information.

4. **Trojan/Backdoor Attack**  
   Inserts a hidden trigger within the model, resulting in a reduced accuracy when the trigger is activated.

---

## Dataset: NSL-KDD

The **NSL-KDD dataset** is a dataset for evaluating intrusion detection on network traffic. It contains labeled instances of normal and abnormal traffic, making it a great choice for analyzing the attacks.

---

## Getting Started

### Prerequisites
- Required libraries: NumPy, requests, pandas, scikit-learn, tensorflow

### Installation
Clone this repository and install the necessary dependencies:
```
git clone <repository_url>
cd <repository_directory>
pip install requests pandas numpy scikit-learn tensorflow
```

---

## Running the Attacks

Each attack has its own file. Follow the instructions below to execute them in the Visual Studio terminal:

1. **DoS Attack**:
   ```
   python "DoS Attack.ipynb"
   ```

2. **Adversarial Attack**:
   ```
   python adversarial_attack.py
   ```

3. **Membership Inference Attack**:
   ```
   python membership_inference_attack.py
   ```

4. **Trojan/Backdoor Attack**:
   ```
   python trojan_attack.py
   ```

The trained models are saved in  `models/` directory.

---

## References

- **NSL-KDD Dataset**: https://www.kaggle.com/datasets/hassan06/nslkdd?select=KDDTest%2B.arff
- All other references are included in the Report
---
