"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.

    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ----------- preprocessing steps --------
    
    important_features = ['Weight_Kg','Low_Price','High_Price','Sales_Total','Total_Qty_Sold','Total_Kg_Sold','Stock_On_Hand','Province_EASTERN CAPE','Province_NATAL','Province_TRANSVAAL','Province_W.CAPE-BERGRIVER ETC','Province_WEST COAST','Container_DT063','Container_EC120','Container_EF120','Container_IA400','Container_JE090','Container_JG110','Container_M4183','Container_M9125','Size_Grade_1M','Size_Grade_1S','Size_Grade_1X','Size_Grade_2L','Size_Grade_2M','Size_Grade_2S','Size_Grade_2U','Date_2020-02-01','Date_2020-02-05','Date_2020-02-14','Date_2020-03-10','Date_2020-03-16','Date_2020-04-17','Date_2020-04-22','Date_2020-04-30','Date_2020-05-22','Date_2020-08-11','Date_2020-08-15','Date_2020-08-18','Date_2020-09-09']
    

    # filter to specific apples
    feature_vector_df = feature_vector_df[feature_vector_df['Commodities'] == 'APPLE GOLDEN DELICIOUS']
    # drop Commodities since we only have 1, and we don't want dummy commodities
    feature_vector_df.drop('Commodities', axis=1, inplace=True)

    # then get dummies on full dataset
    dummy_df = pd.get_dummies(
        feature_vector_df, 
        drop_first=True, 
        columns=['Province', 'Size_Grade', 'Container']
    )

    # helper function:
    clean_name = lambda name: ''.join([c for c in name if c.isalnum() or (c in ['_', ' '])]).replace(' ', '_')
    
    # clean column names
    dummy_df.columns = [clean_name(col) for col in dummy_df.columns] 

    # find which columns are missing in our dummy_df
    missing_cols = set(important_features) - set(dummy_df.columns)
    for mc in missing_cols:
        dummy_df[mc] = 0
    # done
    predict_vector = dummy_df[important_features]
                                
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))

def make_prediction(data, model):
    """Prepare request data for model prediciton.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standerdisation.
    return prediction[0].tolist()
