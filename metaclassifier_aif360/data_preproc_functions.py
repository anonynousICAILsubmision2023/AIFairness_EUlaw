from compas_dataset import CompasDataset
from adult_dataset import AdultDataset
from titanic_dataset import TitanicDataset
import pandas as pd
import numpy as np

def load_preproc_data_compas(label_name='two_year_recid', protected_attributes=None):
    def custom_preprocessing(df):
        """The custom pre-processing function is adapted from
            https://github.com/fair-preprocessing/nips2017/blob/master/compas/code/Generate_Compas_Data.ipynb
        """

        # Restrict races to African-American and Caucasian
        dfcut = df.loc[~df['race'].isin(['Native American','Hispanic','Asian','Other']),:]

        # Restrict the features to use
        dfcutQ = dfcut[['race', 'sex', 'age', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'charge_degree (misd/fel)', 'two_year_recid', 'compas_guess']].copy()

        # Quantize priors count between 0, 1-3, and >3
        def quantizePrior(x):
            if x <=2:
                return '0 to 2'
            elif 2<x<=8:
                return '3 to 8'
            else:
                return 'More than 8'

        def group_race(x):
            if x == "Caucasian":
                return 1.0
            else:
                return 0.0

        dfcutQ['priors_count'] = dfcutQ['priors_count'].apply(lambda x: quantizePrior(x))

        # Recode sex and race
        dfcutQ['sex'] = dfcutQ['sex'].replace({'Female': 1.0, 'Male': 0.0})
        dfcutQ['race'] = dfcutQ['race'].apply(lambda x: group_race(x))

        features = ['race', 'sex', 'age', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'charge_degree (misd/fel)', 'two_year_recid', 'compas_guess']

        # Pass vallue to df
        df = dfcutQ[features]
        target = label_name+' Binary'
        df[target] = df[label_name]

        return df

    XD_features = ['race', 'sex', 'age', 'juv_fel_count', 'juv_misd_count', 'priors_count', 'charge_degree (misd/fel)']
    D_features = ['sex', 'race']  if protected_attributes is None else protected_attributes
    Y_features = [label_name]
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['priors_count', 'charge_degree (misd/fel)']

    # privileged classes
    all_privileged_classes = {"sex": [1.0],
                              "race": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"sex": {0.0: 'Male', 1.0: 'Female'},
                                    "race": {1.0: 'Caucasian', 0.0: 'Not Caucasian'}}


    return CompasDataset(
        label_name=Y_features[0],
        favorable_classes=['No recid.','No recid.'],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        na_values=[],
        metadata={'label_maps': [{0.0: 'Did recid.', 1.0: 'No recid.'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing)

def load_preproc_data_adult(protected_attributes=None, sub_samp=False, balance=False):
    def custom_preprocessing(df):
        """The custom pre-processing function is adapted from
            https://github.com/fair-preprocessing/nips2017/blob/master/Adult/code/Generate_Adult_Data.ipynb
            If sub_samp != False, then return smaller version of dataset truncated to tiny_test data points.
        """

        # Group age by decade
        df['Age (decade)'] = df['age'].apply(lambda x: x//10*10)
        # df['Age (decade)'] = df['age'].apply(lambda x: np.floor(x/10.0)*10.0)

        def group_edu(x):
            if x <= 5:
                return '<6'
            elif x >= 13:
                return '>12'
            else:
                return x

        def age_cut(x):
            if x >= 70:
                return '>=70'
            else:
                return x

        def group_race(x):
            if x == "White":
                return 1.0
            else:
                return 0.0

        # Cluster education and age attributes.
        # Limit education range
        df['Education Years'] = df['education-num'].apply(lambda x: group_edu(x))
        df['Education Years'] = df['Education Years'].astype('category')

        # Limit age range
        df['Age (decade)'] = df['Age (decade)'].apply(lambda x: age_cut(x))

        # Rename income variable
        df['Income Binary'] = df['income-per-year']
        df['Income Binary'] = df['Income Binary'].replace(to_replace='>50K.', value='>50K', regex=True)
        df['Income Binary'] = df['Income Binary'].replace(to_replace='<=50K.', value='<=50K', regex=True)

        # Recode sex and race
        df['sex'] = df['sex'].replace({'Female': 0.0, 'Male': 1.0})
        df['race'] = df['race'].apply(lambda x: group_race(x))

        if sub_samp and not balance:
            df = df.sample(sub_samp)
        if sub_samp and balance:
            df_0 = df[df['Income Binary'] == '<=50K']
            df_1 = df[df['Income Binary'] == '>50K']
            df_0 = df_0.sample(int(sub_samp/2))
            df_1 = df_1.sample(int(sub_samp/2))
            df = pd.concat([df_0, df_1])
        return df

    XD_features = ['Age (decade)', 'Education Years', 'sex', 'race']
    D_features = ['sex', 'race'] if protected_attributes is None else protected_attributes
    Y_features = ['Income Binary']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['Age (decade)', 'Education Years']

    # privileged classes
    all_privileged_classes = {"sex": [1.0],
                              "race": [1.0]}

    # protected attribute maps
    all_protected_attribute_maps = {"sex": {1.0: 'Male', 0.0: 'Female'},
                                    "race": {1.0: 'White', 0.0: 'Non-white'}}

    return AdultDataset(
        label_name=Y_features[0],
        favorable_classes=['>50K', '>50K.'],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        na_values=['?'],
        metadata={'label_maps': [{1.0: '>50K', 0.0: '<=50K'}],
                  'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing)

def load_preproc_data_titanic(protected_attributes=None, sub_samp=False, balance=False):
    def custom_preprocessing(df):

        # everything needs to be numeric
        # does it also have to be binary???
        
        # Group age by decade, and price by 50
        df['Age (decade)'] = df['Age_F'].apply(lambda x: x//10*10)
        df['Price (50)'] = df['Price'].apply(lambda x: x//50*50)

        def age_cut(x):
            if x >= 70:
                return '>=70'
            else:
                return x

        def price_cut(x):
            if x >= 150:
                return '>=150'
            else:
                return x

        def group_class(x):
            if x == 1:
                return 1.0
            else:
                return 0.0

        # Limit ranges
        df['Age (decade)'] = df['Age (decade)'].apply(lambda x: age_cut(x))
        df['Price (50)'] = df['Price (50)'].apply(lambda x: price_cut(x))

        # Recode sex and class
        df['Sex'] = df['Sex'].replace({'female': 1.0, 'male': 0.0})
        df['Class'] = df['Class'].apply(lambda x: group_class(x))

        # Rename income variable
        df['Survived Binary'] = df['Survived']


        if sub_samp and not balance:
            df = df.sample(sub_samp)
        if sub_samp and balance:
            df_0 = df[df['Survived'] == 0]
            df_1 = df[df['Survived'] == 1]
            df_0 = df_0.sample(int(sub_samp/2))
            df_1 = df_1.sample(int(sub_samp/2))
            df = pd.concat([df_0, df_1])
        return df

    XD_features = ['Age (decade)', 'Price (50)', 'Name', 'Sex', 'Class', 'Joined', 'sibsp', 'parch']
    D_features = ['Sex', 'Class'] if protected_attributes is None else protected_attributes
    Y_features = ['Survived Binary']
    X_features = list(set(XD_features)-set(D_features))
    categorical_features = ['Age (decade)', 'Price (50)', 'Name', 'Joined']

    # privileged classes
    all_privileged_classes = {"Sex": [1.0], "Class": [1]}

    # protected attribute maps
    all_protected_attribute_maps = {"Sex": {0.0: 'male', 1.0: 'female'}, "Class": {1.0: 1.0, 0.0: 0.0}}

    return TitanicDataset(
        label_name=Y_features[0],
        favorable_classes=['survived', 'survived'],
        protected_attribute_names=D_features,
        privileged_classes=[all_privileged_classes[x] for x in D_features],
        instance_weights_name=None,
        categorical_features=categorical_features,
        features_to_keep=X_features+Y_features+D_features,
        na_values=['?'],
        metadata={'label_maps': [{1.0: 'survived', 0.0: 'not survived'}],
                'protected_attribute_maps': [all_protected_attribute_maps[x]
                                for x in D_features]},
        custom_preprocessing=custom_preprocessing)
