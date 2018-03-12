import pandas as pd
import os
from os import listdir
from os.path import isfile, join
from itertools import compress
from linear_predictor import XGBPredictor
from tuning import bayesian_optimization
from utils import create_submission


class Ensemble(object):
    def __init__(self, train_y, test_id, tags, data_dir='data/output/'):
        """
        This class creates an ensemble solution from the predictors. The models
        can be ensembled using its mean or XGBoots

        Parameters
        -------------------------------------------------------------------
        train_y: df with the true labels
        test_id: df with the id of the test set
        TAGS: list with the labels names
        data_dir: the path where the models solutions are stored
        """
        self.train_y = train_y
        self.test_id = test_id
        self.TAGS = tags
        self.data_dir = data_dir

    @staticmethod
    def _get_models_name(data_dir):
        """
        Get the folders names where the model values are stored
        """
        call_folders = [x[0] for x in os.walk(data_dir)][1::]
        folder = []
        for i in call_folders:
            temp = i.split('\\')
            folder.append(temp[1])
        print('The models to ensemble are {}'.format(folder))
        return folder

    @staticmethod
    def _get_model_val(models_name, data_dir, val_source='test'):
        """
        Get thr mofrl values labeled with 'val_source'
        """
        model_val = {}
        for model in models_name:
            mypath = data_dir + '/' + model
            only_files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            select_files = [val_source in x for x in only_files]
            only_files = list(compress(only_files, select_files))
            if type(only_files) == list:
                for name_file in only_files:
                    df_name = name_file.replace('.csv', '')
                    model_val[model + '_' + df_name] = pd.read_csv(mypath + '/' + name_file)
            else:
                df_name = only_files.replace('.csv', '')
                model_val[model + '_' + df_name] = pd.read_csv(mypath + '/' + only_files)
        return model_val

    def meta_learner(self, params, predictor=XGBPredictor):
        """
        Ensembler method that uses XGBoost

        Parameters
        -------------------------------------------------------------------
        params: the parameters to tune the predictor
        predictor: the predictor thatis used to ensemble
        """
        models_name = self._get_models_name(self.data_dir)
        if type(models_name) != list:
            models_name = [models_name]

        get_train_y = self._get_model_val(models_name, self.data_dir, 'train')
        get_test_y = self._get_model_val(models_name, self.data_dir, 'test')
        train = pd.DataFrame()
        test = pd.DataFrame()

        for add_train, add_test in zip(list(get_train_y.keys()), list(get_test_y.keys())):
            if train.empty:
                train = get_train_y[add_train][self.TAGS]
                test = get_test_y[add_test][self.TAGS]
            else:
                train = pd.concat([train, get_train_y[add_train][self.TAGS]], axis=1, ignore_index=True)
                test = pd.concat([test, get_test_y[add_test][self.TAGS]], axis=1, ignore_index=True)

        best_params, best_score = bayesian_optimization(predictor, train, self.train_y, params, model_type='GP', acquisition_type='EI',
                                                        acquisition_weight=2, max_iter=10, max_time=None, silent=True, persist=False)

        print('The performance score must be {}'.format(round(best_score, 2)))
        _predictor = predictor(**best_params)
        create_submission(_predictor, train, self.train_y, test, self.train_id, self.test_id,
                          write_to='data/output/submission_{}_ensembler.csv'.format(predictor.name),
                          data_source_nature='sparce_matrix', to_ensamble=False)

    def mean_ensembler(self):
        """
        Ensembler method that uses the average

        Parameters
        -------------------------------------------------------------------
        params: the parameters to tune the XGBoost algorithm
        """
        models_name = self._get_models_name(self.data_dir)
        if type(models_name) != list:
            models_name = [models_name]

        get_test_y = self._get_model_val(models_name, self.data_dir, 'test')
        # Calculate the average
        average = pd.DataFrame()
        for add in list(get_test_y.keys()):
            if average.empty:
                average[self.TAGS] = get_test_y[add][self.TAGS]
            else:
                average[self.TAGS] += get_test_y[add][self.TAGS]

        average = average/len(list(get_test_y.keys()))
        average.insert(loc=0, column='id', value=self.test_id.values)
        doc_name = self.data_dir + '/' + 'submission_average_ensembler.csv'
        average.to_csv(doc_name, index=False)
        print('submission file saved at {}'.format(doc_name))
        return average
