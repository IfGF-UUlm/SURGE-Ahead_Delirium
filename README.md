# Post-Operative Delirium Prediction Model
This GitHub repository contains the source code for our explainable machine learning (ML) model predicting post-operative delirium (POD) in older patients undergoing surgery. The project was developed as part of the SURGE-Ahead (Supporting SURgery with GEriatric Co-Management and AI) project to create a digital healthcare application for assisting surgical teams in caring for geriatric patients.

## Project Overview
Our study aims to develop an algorithm that is robust across clinical settings, highly automated, and straightforwardly explainable. We limited our model selection to linear ML models (logistic regression and linear support vector machine) due to transparency and robustness considerations. Our final ML model, a linear SVM, uses 15 features from the PAWEL dataset, including age, surgery type, ASA score, use of cardio-pulmonary bypass, clinical frailty score, cut-to-suture time, preexisting dementia, eGFR, MoCA subscores (memory, orientation, and verbal fluency), polypharmacy, multimorbidity, postoperative isolation, pre-operative benzodiazepines, recent falls, and number of medications.

## Project Structure
The repository has the following structure:
- [`LICENSE`](./LICENSE): MIT License for this project
- [`LR.py`](./LR.py): Implementation of the logistic regression (LR) model
- [`SVM.py`](./SVM.py): Implementation of the linear support vector machine (SVM) model
- [`main.py`](./main.py): Main script to train the SVM model and evaluate its performance
- [`requirements.txt`](./requirements.txt): List of required Python packages for this project
- [`utils.py`](./utils.py): Utility and Helper Functions

## Getting Started
This repository reports the code for transparency. As soon as the PAWEL dataset is released to the public, the data needed to train the models will be added to the repository. To inspect and review the code:

1. Clone this repository to your local machine using `git clone https://github.com/IfGF-UUlm/SURGE-Ahead_Delirium.git`.

As soon as the dataset is released you may also:

3. Set up a virtual enviroment by running `python3 -m venv env` and activate it.
4. Install required dependencies by running `pip install -r requirements.txt`.
5. Run the main script with `python main.py` to train and evaluate the SVM model.
6. If you want to train and evaluate the LR model, simply import the LR instead of the SVM module in `main.py`.

## Dependencies
This project requires Python 3.x and the following packages:
- NumPy
- Pandas
- Scikit-learn

You can install all dependencies by running `pip install -r requirements.txt`.

## License
This project is licensed under the [MIT License](./LICENSE).

## Citation
If you use this code in your research, please cite our paper: \
[The associated paper has been submitted for publication. The citation details will be updated once the paper is accepted.]

## Contact
For any questions, feedback, or concerns, please contact us at [thomas.kocar@uni-ulm.de](mailto:thomas.kocar@uni-ulm.de).
