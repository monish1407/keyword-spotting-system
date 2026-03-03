\# 🎙️ Keyword Spotting System (LSTM)



A PyTorch-based speech command classification system that detects spoken keywords using MFCC features and an LSTM neural network.



---



\## 🚀 Project Overview



This project implements a \*\*Keyword Spotting System\*\* trained on the Google Speech Commands dataset.



The model classifies 10 spoken commands:



```

yes, no, up, down, left, right, on, off, stop, go

```



The system converts audio signals into MFCC features and feeds them into an LSTM model for sequence classification.



---



\## 🧠 Model Architecture



\- \*\*Feature Extraction\*\*: 40 MFCC coefficients

\- \*\*Sequence Length\*\*: 32 time steps

\- \*\*Model\*\*: LSTM (Hidden Size: 192)

\- \*\*Dropout\*\*: 0.3

\- \*\*Optimizer\*\*: Adam

\- \*\*Loss Function\*\*: CrossEntropyLoss

\- \*\*Epochs\*\*: 25

\- \*\*Batch Size\*\*: 64



---



\## 📊 Results



| Metric | Value |

|--------|--------|

| Test Accuracy | \*\*78.68%\*\* |

| Classes | 10 |

| Samples per Class | 300 |

| Device | CPU |



---



\## 📂 Project Structure



```

keyword-spotting-system/

│

├── src/

│   ├── config.py        # Hyperparameters and settings

│   ├── features.py      # MFCC feature extraction

│   ├── model.py         # LSTM model definition

│   └── train.py         # Training and evaluation script

│

├── data/                # Dataset (auto-downloaded, ignored by git)

├── models/              # Saved model weights (ignored by git)

├── requirements.txt     # Project dependencies

└── README.md

```



---



\## ⚙️ Installation



Clone the repository:



```bash

git clone https://github.com/monish1407/keyword-spotting-system.git

cd keyword-spotting-system

```



Create a virtual environment (Python 3.11 recommended):



```bash

py -3.11 -m venv venv

venv\\Scripts\\activate

```



Install dependencies:



```bash

pip install -r requirements.txt

```



---



\## ▶️ Train the Model



Run the training script:



```bash

python src/train.py

```



\- The dataset will download automatically.

\- After training, the model will be saved in the `models/` folder.



---



\## 💾 Model Output



Trained model weights are saved as:



```

models/keyword\_model.pth

```



---



\## 🎯 Key Learning Outcomes



\- Audio preprocessing using MFCC

\- Sequence modeling with LSTM

\- Dataset filtering and class balancing

\- Model training and evaluation

\- Dependency management with virtual environments

\- Clean project structuring for GitHub



---



\## 👨‍💻 Author



\*\*Monish Patil\*\*  

GitHub: https://github.com/monish1407

