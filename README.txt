Description :
- Dashboard explaining the decision taken by a classifier.
- Contains feature importance, SHAP explanation waterfall, close individuals 3d graph and timeline of same id individuals.

Versions : 
- Python 3.8
- See requirements.txt for the rest

Config usage :
- One config corresponds to multiple models and datasets. All combinations of models and datasets withing a config should be compatible, otherwise create another config.
- The folder names are the IDs of the configs or the datasets, the filenames (config.json, data.csv) can't change.
- The models are stored without IDs, their filename identify them in a config.

- Models are loaded using pickle
- Datasets are loaded using pandas

    General config :
        - (List[str]) models : List of the models filenames for this config.
        - (List[str]) datasets : List of the datasets ids for this config.

    Dataset config :
        - display_fields : The fields that will be displayed when selecting an operation ID.
        - columns_config : 
            - (str) class : Column name that contains the prediction {0, 1}.
            - (str) id : Any column containing a unique ID. (int, str)
            - (str) prediction_score : Column containing the prediction scores [0, 1].
            - (str) time : Column containing the operation time. This time will be converted with pd.to_datetime().
            - (str) quantity : Column containing a form of quantity [-inf, inf]. Eg : Amount spent in one transaction
            - (List[str]) features : List containing the exact feature names used to train the model.

Dashboard features :
