from sklearn.base import BaseEstimator, TransformerMixin
from .pickleable_model import Pickleable_Model
import h2o
import pandas as pd

class Custom_Estimator(BaseEstimator, TransformerMixin):
    '''
    Custom_Estimator wraps the h2o model as a standard scikit-learn estimator.
    However, only the function `predict()` is implemented in this stage.
    H2O model has to be trained first in a standard way and later be wrapped into
    the Custom_Estimator with:
    >>  Custom_Estimator(model_id, custom_types).
    '''
    def __init__(self, model_id, column_types= {} ):
        super().__init__()
        # Transform the model into pickleable model.
        self.model_id = model_id
        self.column_types = column_types
        self.pic_model = Pickleable_Model(model_id)
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        return self.predict(X)

    def predict(self, X):
        data_df = X

        # Upload to h2o server.
        data_hdf = h2o.H2OFrame(data_df, column_types = self.column_types)

        # Make prediction
        result_hdf = self.pic_model.predict(data_hdf)
        result_hdf = data_hdf.cbind(result_hdf)

        # Transform back to pandas data frame.
        result_df = result_hdf.as_data_frame()

        # Release memory
        h2o.remove(result_hdf)

        return result_df
  