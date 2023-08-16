#!/usr/bin/env python
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegressionCV

# Helper function for always rounding up a 5
def normal_round(x, digits=0):
    x += (x % (1 ** (-digits)) == 0.5 * 10 ** (-digits)) * (0.1 * 10 ** (-digits))
    return np.round(x, digits)

class DeliriumPredictor():
    def __init__(self):
        self._cont_vars = [
            'Cut-to-Suture Time',
            'Age',
            'GFR (Cockcroft-Gault)',
            'Hemoglobin',
        ]
        self._ord_vars = [
            'ASA Class',
            'MoCA Orientation',
            'MoCA Memory',
            'Number of Medications',
            'Multimorbidity',
            'Clinical Frailty Scale',
        ]
        self._bin_vars = [
            'MoCA Verbal Fluency',
            'Dementia',
            'Recent Fall',  
            'Post-OP Isolation',
            'Pre-OP Benzodiazepines',
            'Cardio-Pulmonary Bypass',
            # 'Anamnese_Delir',
        ]

    def _load_data(self, filename, testdata=None):
        """Load training data from the given file. Isolate the target variable.
        The traing and test sets can come from two different files. In that case
        the first argument is the traing data and the second argument the test data"""
        data = pd.read_csv(filename)
        target = data['Delirium']
        data = data[self._cont_vars + self._ord_vars + self._bin_vars]

        if testdata is not None:
            self.X_train = data
            self.y_train = target
            self.X_test = pd.read_csv(testdata)
            self.y_test = self.X_test['Delirium']
            self.X_test = self.X_test[self._cont_vars + self._ord_vars + self._bin_vars]
        else:
            (self.X_train,
             self.X_test,
             self.y_train,
             self.y_test) = train_test_split(
                data,
                target,
                test_size=0.2,
                random_state=30598
            )

    def _make_preprocessor(self):
        """Constructs a preprocessing pipeline for the data.
        For continuous variables, missing values are replaces by the
        mean, rounded to whole numers. The values then undergo Z-transformation.
        Ordinal variables are imputed usign the median and then Z-transformed.
        Binary variables are Imputed using the most frequent value.

        Returns the preprocessing pipeline as a
        sklearn.compose.ColumsTransformer object."""
        cont_pipe = make_pipeline(
            SimpleImputer(strategy='mean'),
            StandardScaler(),
        )
        ord_pipe = make_pipeline(
            SimpleImputer(strategy='median'),
            StandardScaler(),
        )
        bin_pipe = make_pipeline(
            SimpleImputer(strategy="most_frequent"),
            StandardScaler(with_mean=False, with_std=False),
        )

        self.transformer = make_column_transformer(
            (cont_pipe, self._cont_vars),
            (ord_pipe, self._ord_vars),
            (bin_pipe, self._bin_vars),
            verbose_feature_names_out=False,
        ).set_output(transform='pandas')

        # Round all parameters. Imputed values to integers, scaling parameters
        # to 2 decimal places.
        self.transformer.fit(self.X_train)
        for pipeline in self.transformer.transformers_:
            si = pipeline[1].named_steps['simpleimputer']
            ss = pipeline[1].named_steps['standardscaler']
            si.statistics_ = normal_round(si.statistics_)
            ss.mean_ = normal_round(ss.mean_, 2) if ss.mean_ is not None else None
            ss.scale_ = normal_round(ss.scale_, 2) if ss.scale_ is not None else None

        return self.transformer

    def _preprocess(self):
        """Preprocesses the data using the trained transformer.

        Returns imputed and scaled X_train, y_train, X_test, y_test"""
        self.X_train = self.transformer.transform(self.X_train)
        self.X_test = self.transformer.transform(self.X_test)

        return self.X_train, self.y_train, self.X_test, self.y_test

    def _train(self):
        """Tranins a Logistic Regression, finding the optimal value of
        the regularisation hyperparameter C using a LeaveOneOut cross-
        validation. All coefficients are rounded to two decimal places.

        Returns the trained Logistic Regression Classifier, which is a
        sklearn.linear_models.LogisticRegressionCV object."""
        self.lr = LogisticRegressionCV(
            max_iter=int(1e9),
            Cs=np.logspace(-8, 8, 17, base=2),
            cv=LeaveOneOut(),
            solver='liblinear',
            n_jobs=-1,
            verbose=0,
        ).fit(self.X_train, self.y_train)
        self.lr.coef_ = normal_round(self.lr.coef_, 2)
        self.lr.intercept_ = normal_round(self.lr.intercept_, 2)

        return self.lr

    def predict(self, x, preprocessed=True):
        """Predicts the outcome for a given X. If X isn't already preprocessed, the
        preprocessing pipeline trained in self.preprocess() is used on the data.

        Returns a tuple of (binary prediction, predicted probability of True).

        The input X can be a two-dimensional array containing multiple datapoints.
        In that case the return value is a tuple of two arrays."""
        if not preprocessed:
            x = self.transformer.transform(x)
        return (self.lr.predict(x), self.lr.predict_proba(x)[:, 1])

    def store_model(self, filename):
        """stores the fitted self.transformer and self.lr as a pickled file"""
        with open(filename, 'wb') as f:
            pickle.dump((self.transformer, self.lr), f)

    @classmethod
    def from_pickle(cls, filename):
        """loads a fitted self.transformer and self.lr from a pickled file"""
        res = cls()
        with open(filename, 'rb') as f:
            res.transformer, res.lr = pickle.load(f)
            return res

    @classmethod
    def from_dataset(cls, filename, testdata=None):
        """Creates a class instance that loads the data fromt he given file(s),
        preprocesses it, and uses it to train  the model"""
        res = cls()
        print("Loading data")
        res._load_data(filename, testdata)
        print("Preprocessing")
        res._make_preprocessor()
        res._preprocess()
        print("Training model")
        res._train()
        return res


if __name__ == "__main__":
    print(
        "Import this file as a module.\n",
        "The class 'DeliriumPredictor()' has the methods\n",
        "'load_data(filename)' for loading training data,\n",
        "'preprocess()' for creating a preprocessing pipeline,\n",
        "'tain()' for training the model, and \n",
        "'predict()' for calculating predictions based on input data.",
        sep=''
    )
