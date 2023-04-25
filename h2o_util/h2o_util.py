import os
import subprocess
import shutil
import pandas as pd
import random
import tempfile
from .H2O_Pickle_Model import H2O_Pickle_Model
import h2o
from h2o.automl import H2OAutoML 
from sklearn.model_selection import train_test_split
import typing

class H2O_Util:
    """
    Utilities to simplify H2O module.
    """

    def __init__(
        self,
        column_types: typing.Dict[str, str] = {},
        x_cols: typing.List[str] = [],
        y_col: str = '',
        rename_prediction_col_name: typing.Dict[str, str] = {},
        predict_col_name = 'predict',
        proba_col_name = 'proba',
        automl_config: typing.Dict[str, str] = {},
        init_dict: typing.Dict[str, str] = {},
    ):
        """
        Initializes an instance of H2O_Util.

        Args:
            column_types (Dict[str, str], optional): A dictionary mapping column names to their types. Defaults to {}.
            x_cols (List[str], optional): A list of column names to use as features. Defaults to [].
            y_col (str, optional): The name of the column to use as the target. Defaults to ''.
            rename_prediction_col_name (Dict[str, str], optional): A dictionary mapping column names to their new names. Defaults to {}.
            predict_col_name (str, optional): The name of the column containing predicted values. Defaults to 'predict'.
            proba_col_name (str, optional): The name of the column containing predicted probabilities. Defaults to 'proba'.
            automl_config (Dict[str, str], optional): A dictionary of configuration options for AutoML. Defaults to {}.
            init_dict (Dict[str, str], optional): A dictionary of configuration options for H2O. Defaults to {}.
        Returns:
            None
        """
        self.column_types = column_types
        self.x_cols = x_cols
        self.y_col = y_col
        self.automl_config = automl_config
        self.rename_prediction_col_name = rename_prediction_col_name
        self.predict_col_name = predict_col_name
        self.proba_col_name = proba_col_name

        # Connect to H2O.
        h2o.init(**init_dict)        
    
    def serialize_model(self, model_id: str) -> bytearray:
        """
        Serialize the h2o model into binary array.

        Args:
            model_id: The name of the model. Assume model in H2O server.

        Returns:
            The model in binary array.
        """

        # Create temporary folder
        temp_path = tempfile.TemporaryDirectory()

        # Save model.
        model = h2o.get_model(model_id)
        filename = h2o.save_model(model, temp_path.name, force=True)

        # Read back the model from disk
        with open(filename, 'rb') as f:
            bin_file = f.read()

        # Clear the temporary folder
        temp_path.cleanup()

        return bytearray(bin_file)
    

    def unserialize_model(self, model_barray: bytearray) -> h2o.estimators:
        """
        Unserialize a binary array into h2o model.

        Args:
            model_barray: Model in bytearray.

        Returns:
            Model object reference to H2O server.
        """

        # Create temporary folder
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(model_barray)
            model = h2o.load_model(tmp.name)

        return model
    
    def save_model(self, model: h2o.estimators, model_dir_path: str) -> None:
        """
        Save the h2o model to the specified directory.

        Args:
            model: The h2o model to save.
            model_dir_path: The directory to save the model to.

        Raises:
            ValueError: If the model_dir_path already exists and contains more than one file.
        """

        # Check if the model_dir_path already exists.
        # if os.path.isdir(model_dir_path):
        #     # Check if the model_dir_path contains more than one file.
        #     if len(os.listdir(model_dir_path)) > 1:
        #         raise ValueError(
        #             f"The model_dir_path '{model_dir_path}' already exists and contains more than one file."
        #         )

        # Save the model.
        h2o.save_model(model, model_dir_path, force=True)


    def load_model(self, model_dir_path: str) -> h2o.estimators:
        """
        Load the h2o model from the specified directory.

        Args:
            model_dir_path: The directory to load the model from.

        Returns:
            The loaded h2o model.

        Raises:
            ValueError: If the model_dir_path does not exist or does not contain a model.
        """

        # Check if the model_dir_path exists.
        if not os.path.isdir(model_dir_path):
            raise ValueError(f"The model_dir_path '{model_dir_path}' does not exist.")

        # Check if the model_dir_path contains a model.
        if len(os.listdir(model_dir_path)) == 0:
            raise ValueError(f"The model_dir_path '{model_dir_path}' does not contain a model.")

        # Load the model.
        model_path = os.path.join(model_dir_path, os.listdir(model_dir_path)[0])
        model = h2o.load_model(model_path)

        return model
            
    def sample(self, data_hdf: h2o.H2OFrame, nsample: int) -> h2o.H2OFrame:
        """
        Select random rows from h2o data frame.

        Args:
            data_hdf: The h2o data frame.
            nsample: The number of random sample.

        Returns:
            H2O data frame.
        """

        nrow = data_hdf.nrow
        select_list = random.choices(range(nrow), k=nsample)
        select_list = sorted(select_list)
        return data_hdf[select_list, :]


    def to_h2o_data_frame(self, pandas_data_frame: pd.DataFrame, column_types: dict = {}) -> h2o.H2OFrame:
        """
        Safely convert pandas and modin data frame into H2O data frame.

        Args:
            pandas_data_frame: Pandas data frame.
            column_types: Type of the columns in dict.

        Returns:
            H2O data frame.
        """

        return h2o.H2OFrame(pandas_data_frame, column_types=column_types)


    def predict_model(
        self,
        model: h2o.estimators,
        data: pd.DataFrame,
        return_full_result: bool = True,
    ) -> pd.DataFrame:
        """
        Make prediction.

        Args:
            model: h2o model.
            data: Pandas data frame to make prediction.
            return_full_result: If "default", use the default h2o_util_dict. Else, use the specify one.

        Returns:
            Predicted output in pandas data frame.
        """

        data_hdf = self.to_h2o_data_frame(data, self.column_types)

        result_hdf = model.predict(data_hdf)

        if return_full_result:
            result_hdf = data_hdf.cbind(result_hdf)

        result_df = result_hdf.as_data_frame()
        result_df.index = data.index.copy()  # reset the index

        # Rename columns
        result_df = result_df.rename(columns=self.rename_prediction_col_name)

        # Release
        h2o.remove(result_hdf)
        h2o.remove(data_hdf)

        return result_df

    def split_data(
        self,
        data: pd.DataFrame,
        test_size: float = 0.3,
        random_state: typing.Optional[int] = None,
    ) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split the data to train and test.

        Args:
            data_df: The data frame to split.
            test_size: The amount to allocate to test data.
            random_state: Optional random number.

        Returns:
            Train data frame.
            Test data frame.
        """
        if random_state is None:
            random_state = random.randrange(10**8)
        return train_test_split(data, test_size=test_size, random_state=random_state)


    def model_performance(
        self,
        h2o_model,
        test_df: pd.DataFrame,
        thresholds: float = 0.5,
    ) -> typing.Dict[str, float]:
        """
        Summarize the model performance.

        Args:
            h2o_model: H2O model.
            test_df: Test data frame.
            threshold: Prediction threshold.

        Returns:
            Performance in dict.
        """
        test_hdf = self.to_h2o_data_frame(test_df, self.column_types)

        result_dict = {}

        performance = h2o_model.model_performance(test_hdf)

        result_dict["accuracy"] = performance.accuracy(thresholds=thresholds)[0][1]
        result_dict["f1"] = performance.F1(thresholds=thresholds)[0][1]
        result_dict["auc"] = performance.auc()
        result_dict["precision"] = performance.precision(thresholds=thresholds)[0][1]
        result_dict["recall"] = performance.recall(thresholds=thresholds)[0][1]
        result_dict["confusion_matrix"] = performance.confusion_matrix(thresholds=[thresholds])

        # Release
        h2o.remove(test_hdf)

        return result_dict

    def train_model_with_automl(
        self,
        data: pd.DataFrame,
    ) -> tuple:
        """
        Train model with H2O automl

        Args:
            data_df: Pandas's data frame.
            x_cols: The feature columns.
            y_col: The target column.
            column_types: The type of the columns in data_df.

        Returns:
            h2o model.
            h2o leaderboard.
        """

        column_types = self.column_types
        x_cols = self.x_cols
        y_col = self.y_col

        # Upload the data frame to h2o.
        data_hdf = self.to_h2o_data_frame(data, column_types=column_types)

        # Train model
        aml = H2OAutoML(**self.automl_config)
        aml.train(
            x=x_cols,
            y=y_col,
            training_frame=data_hdf,
        )

        # Release
        h2o.remove(data_hdf)

        return aml.leader, aml.leaderboard.as_data_frame()


    def to_pickable_model(self, model: h2o.estimators) -> H2O_Pickle_Model:
        """
        Convert a h2o model to a h2o pickable model.
        The model data will be auto uploaded to h2o server when using.

        Args:
            model: H2O model.

        Returns:
            pickable model.
        """
        pmodel = H2O_Pickle_Model(model.model_id)
        return pmodel


    def clean_temp(self, ndays: int = 10) -> int:
        """
        Clean temp files that are ndays older.
        Use in Linux os with care.

        Args:
            ndays: Number of days.

        Returns:
            Return code. 0 for success and otherwise.
        """
        cmd = f'find /tmp -type f -atime +{ndays} -delete'
        cmd_list = cmd.split()
        result = subprocess.run(cmd_list)
        return result.returncode
