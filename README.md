# iris-classifier

A simple machine learning project that trains a Decision Tree classifier on the Iris dataset using scikit-learn.

## Project Structure

iris-classifier/
├── data/
├── notebooks/
│   └── iris/
│       └── _model.ipynb
├── outputs/
├── src/
│   └── _train.py
├── tests/
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── structure.txt

## Quick Start

Clone the repository:

git clone https://github.com/Boritz/iris-classifier.git
cd iris-classifier

Create and activate a virtual environment.

Windows

python -m venv .venv
.venv\Scripts\activate

macOS / Linux

python -m venv .venv
source .venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Run the training script:

python src/_train.py

Output file:

outputs/confusion_matrix.png

Open the notebook:

jupyter notebook

Then open:

notebooks/iris/_model.ipynb