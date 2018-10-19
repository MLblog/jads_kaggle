import pandas as pd


def create_submission(predictor_cls, params, train_x, train_y, test_x, test_ids,
                      save=True, path="../submissions/submission.csv"):
    """ Create a submission file.

    Parameters
    ----------
    predictor_cls: class, inherits from Predictor
        The model you want to use to create the submission.
    params: dict
        The parameters of predictor_cls to create the submission with.
    train_x: pd.DataFrame
        The training data features.
    train_y: array-like
        The target variable of the training data.
    test_x: pd.DataFrame
        The test data features.
    test_ids: array-like
        The fullVisitorIds corresponding to the test features. These are needed to
        create the submission file.
    save: boolean
        Whether to save the submission file as csv or just to return it as a DataFrame.
    path: str
        The path to where the submission should be saved. Ignored if save==False.

    Returns
    -------
    A pd.DataFrame with two columns ['fullVisitorId', 'PredictedLogRevenue']. This file
    can be saved as csv (data.to_csv(path, index=False)) and then uploaded to Kaggle.
    """

    assert len(test_ids) == 617242, \
        "test_ids contains {} rows, while 617,242 are expected.".format(len(test_ids))

    print("Fitting model...")
    model = predictor_cls(**params)
    model.fit(train_x, train_y)
    print("Predicting...")
    test_y = model.predict(test_x)
    print("Creating submission...")
    submission = pd.concat([test_ids, pd.Series(test_y, name="PredictedLogRevenue")], axis=1)

    if save:
        print("Saving submission file...")
        try:
            submission.to_csv(path, index=False)
        except IOError:
            print("Path not found, saving in working directory instead under name 'submission.csv'.")
            submission.to_csv("submission.csv", index=False)

    return submission
