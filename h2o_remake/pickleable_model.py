#
import tempfile
import h2o

class Pickleable_Model():
    '''
    Pickleable_Model is a class that wraps the H2O model such that it can be serialized with Pickle.

    Example:
    To pickle a model, 
    >> model # Assume that this model is trained with H2O.
    >> pic_model = Pickleable_Model(model)
    >> with open('./test.pickle', 'wb') as f:
    ...    picke.dump(pic_model, f)
    ...
    To unpickle a model, 
    >> with open('./test.pickle', 'rb') as f:
    ...    new_model = picke.dump(f)
    ... 

    The existing functions of h2o model is not affected. We can still access those function directly
    such as calling:
    >> pic_model.auc()
    >> pic_model.model_performance()
    '''
    def __init__(self, model_id):
        self.model_id = model_id
        #
        self.model = h2o.get_model(model_id)
        # Load the bytes of the model into memory.
        self.model_bytes_arr = self.serialize_model(self.model_id)
    
    def __getattr__(self, attrname):
        '''Those not existing function will refer back to model'''
        return self.model.__getattribute__(attrname)
    
    def __getstate__(self):
        __dict__ = self.__dict__.copy()
        
        del __dict__['model']
        
        return __dict__
    
    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)
        self.reload_model_to_server()
        
    
    def reload_model_to_server(self):
        try: 
            test_model = h2o.get_model(self.model_id)
        except h2o.exceptions.H2OResponseError:
            if 'model_bytes_arr' in self.__dict__:
                self.unserialize_model(self.model_bytes_arr)
        self.model = h2o.get_model(self.model_id)
        
    
    def serialize_model(self, model_id):
        '''Serialize the h2o model into binary array. '''

        # Create temporary folder
        temp_path = tempfile.TemporaryDirectory()
        
        # Save model.
        model = h2o.get_model(model_id) 
        filename = h2o.save_model(model, temp_path.name, force = True)

        # Read back the model from disk
        with open(filename, 'rb') as f:
            bin_file = f.read()

        # Clear the temporary folder
        temp_path.cleanup()

        return bytearray(bin_file)
    
    def unserialize_model(self, model_barray):
        '''Unserialize a binary array into h2o model.'''
        # Create temporary folder
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(model_barray)
            model = h2o.load_model(tmp.name)
        #
        return model
    
    def __str__(self):
        return 'Represent "%s" with Dapau_H2O_Model.' % self.model_id 