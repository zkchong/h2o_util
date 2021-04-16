from .pickleable_model import Pickleable_Model
from .custom_estimator import Custom_Estimator
class H2O_Remake():
    @classmethod 
    def make_pickleable_model(self, model_id):
        return Pickleable_Model(model_id)
    
    @classmethod
    def make_custom_estimator(self, model_id, column_types = {}):
        # raise ValueError('Not implemented yet')
        return Custom_Estimator(model_id, column_types)
    