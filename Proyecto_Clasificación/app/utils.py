from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.label_encoders = {col: LabelEncoder().fit(X[col]) for col in X.columns}
        return self

    def transform(self, X):
        return X.apply(lambda col: self.label_encoders[col.name].transform(col))

