from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

class CustomLabelEncoder(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelEncoder(*args, **kwargs)
    
    def fit(self, X, y=0):
        self.encoder.fit(X)
        return self
        
    def transform(self, X, y=0):
        return self.encoder.transform(X)
    
