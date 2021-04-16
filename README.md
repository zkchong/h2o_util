# H2O Remake
A library that enables h2o model to be serializable in pickle.

## Description
For those that use H2O (https://www.h2o.ai/) to train a model, we always challenged by the hassle to wrap the h2o model with scikit-learn pipeline (https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#). 

If we just putting the model as an object in the pipeline and save with pickle, the unpickle process will fail.The h2o model variable in our python interpreter is just a link to call the actual h2o model in the h2o server and pickle does not take that into account. We should serialize the model as byte array during Pickle's serialization process. 

Our code `H2O Remake` will wrap the h2o model in a way it can be put into pipeline directly. Note that the h2o model should be pretrained and `fit()` is not implemented. 

## Example
Assume that we have a H2O model named `h2o_model` and a scikit-learn scaler, named `scaler`.
```python
from h2o_remake import H2O_Remake as hr
h2o_estimator = h2o_estimator = hr.make_custom_estimator(h2o_model.model_id)

steps=[ ('scaler', scaler),
        ('h2o_est', h2o_estimator),
      ]

pipeline = Pipeline(steps)

pipeline.predict(test_df) 

```
The detail example code can be found at `Test Custom Transfomer.ipynb`.