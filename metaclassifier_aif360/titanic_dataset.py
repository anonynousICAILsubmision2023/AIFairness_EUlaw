import os

import pandas as pd

from standard_dataset import StandardDataset

default_mappings = {
    'label_maps': [{1.0: 'survived', 0.0: 'not survived'}],
    'protected_attribute_maps': [{0.0: 'male', 1.0: 'female'}, {1.0: 1.0, 0.0: 0.0}]
}

def default_preprocessing(df):
    """Perform the same preprocessing as the original analysis:
    Not decided how best to preprocess yet
    """
    return df

class TitanicDataset(StandardDataset):
    """Titanic Dataset.
    See :file:`aif360/data/raw/compas/README.md`.
    """

    def __init__(self, label_name='Survived', favorable_classes=['survived', 'survived'],
                 protected_attribute_names=['Sex', 'Class'],
                 privileged_classes=[['female'], [1]],
                 instance_weights_name=None,
                 categorical_features=['Name', 'Joined'],
                 features_to_keep=[], # this is empty in adult but full in compas, maybe there's somethig wrong there
                 features_to_drop=[], na_values=[],
                 custom_preprocessing=default_preprocessing,
                 metadata=default_mappings):
        """See :obj:`StandardDataset` for a description of the arguments.
        Examples:
            In some cases, it may be useful to keep track of a mapping from
            `float -> str` for protected attributes and/or labels. If our use
            case differs from the default, we can modify the mapping stored in
            `metadata`:
            >>> label_map = {1.0: 'Did recid.', 0.0: 'No recid.'}
            >>> protected_attribute_maps = [{1.0: 'Male', 0.0: 'Female'}]
            >>> cd = CompasDataset(protected_attribute_names=['sex'],
            ... privileged_classes=[['Male']], metadata={'label_map': label_map,
            ... 'protected_attribute_maps': protected_attribute_maps})
            Now this information will stay attached to the dataset and can be
            used for more descriptive visualizations.
        """

        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '..', 'data', 'raw', 'compas', 'compas-scores-two-years.csv')

        try:
            path = '/content/drive/MyDrive/PhD/Bias detection & mitigation/Legal principles and AI/Lisa/Datasets/'
            df = pd.read_csv(path+'titanic.csv', index_col=0)
            
        except IOError as err:
            print("IOError: {}".format(err))
            print("To use this class, please download the following file:")
            print("\n\tadd link")
            print("\nand place it, as-is, in the folder:")
            print("\n\t{}\n".format(os.path.abspath(os.path.join(
               os.path.abspath(__file__), '..', '..', 'data', 'raw', 'compas'))))
            import sys
            sys.exit(1)

        super(TitanicDataset, self).__init__(df=df, label_name=label_name,
            favorable_classes=favorable_classes,
            protected_attribute_names=protected_attribute_names,
            privileged_classes=privileged_classes,
            instance_weights_name=instance_weights_name,
            categorical_features=categorical_features,
            features_to_keep=features_to_keep,
            features_to_drop=features_to_drop, na_values=na_values,
            custom_preprocessing=custom_preprocessing, metadata=metadata)