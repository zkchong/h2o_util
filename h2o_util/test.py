# import h2o_util
from . import H2O_Util
import h2o
from sklearn import datasets
import pickle

def test():
    """
    Test the H2O_Util class.

    Args:
        None.

    Returns:
        None.
    """
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Import data set
    logging.info('Loading iris data set ...')
    iris = datasets.load_iris(as_frame=True)
    iris_df = iris.frame

    # We only take binary classification for now.
    mask = iris_df.target.isin([0, 1])
    iris_df = iris_df[mask]


    # Identify x_cols, and y_col
    x_cols = [c for c in iris_df.columns if c != 'target']
    y_col = 'target'

    # Setup the column types
    type_dict = {x: 'real' for x in x_cols}
    type_dict[y_col] = 'enum'
    column_types = type_dict

    # Initialize H2O Util.
    logging.info('Initializing H2O_Util ...')
    h2o_util = H2O_Util(
        x_cols=x_cols,
        y_col=y_col,
        column_types=column_types,
        rename_prediction_col_name={
            'predict': 'predict', 
            '1.0': 'proba'
        },
        proba_col_name='proba',
        predict_col_name='predict',
        automl_config={'include_algos': ['GLM']}
    )

    # Split data frame.
    train_df, test_df = h2o_util.split_data(iris_df, 0.3)
    logging.info('Example of Iris data frames:')
    logging.info(iris_df.sample(3))
    logging.info(f'train_df.shape={train_df.shape}')
    logging.info(f'test_df.shape={test_df.shape}')



    # Train H2O automl
    logging.info('Train model with automl ...')
    # automl_dict = {'include_algos': ['GLM'] }

    model, leaderboard = h2o_util.train_model_with_automl(
        data=train_df,
    )

    logging.info('Examine the model performance ...')
    performance_df = h2o_util.model_performance(model, test_df)
    logging.info(performance_df)

    #
    # Serial and Unserialize
    #
    logging.info('Serialize model ... ')
    model_bin = h2o_util.serialize_model(model.model_id)

    # Remove the model from server
    h2o.remove(model.model_id)

    logging.info('Unserialize model ... ')
    new_model = h2o_util.unserialize_model(model_bin)

    # Save model
    logging.info('Save model ... ')
    h2o_util.save_model(new_model, '/tmp/test')

    # Load model
    logging.info('Load model ... ')
    new_model2 = h2o_util.load_model('/tmp/test')

    #
    # To pickable model
    #
    logging.info('Transfer to pickable model ... ')
    pmodel = h2o_util.to_pickable_model(new_model2)

    with open('/tmp/test/pmodel.pickle', 'wb') as f:
        pickle.dump(pmodel, f)

    with open('/tmp/test/pmodel.pickle', 'rb') as f: 
        new_pmodel = pickle.load(f)

    pred_df = h2o_util.predict_model(new_pmodel, test_df)
    print(pred_df.sample(3))
