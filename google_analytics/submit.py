import pandas as pd


def create_submission(predictor_cls, params, train_x, train_y, test_x,
                      save=True, path="../submissions/submission.csv",
                      path_sample_submission="../data/sample_submission.csv"):
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
    save: boolean
        Whether to save the submission file as csv or just to return it as a DataFrame.
    path: str
        The path to where the submission should be saved. Ignored if save==False.
    path_sample_submission: str
        The path where the sample submission file is saved.

    Returns
    -------
    A pd.DataFrame with two columns ['fullVisitorId', 'PredictedLogRevenue']. This file
    can be saved as csv (data.to_csv(path, index=False)) and then uploaded to Kaggle.
    """

    try:
        sample_submission = pd.read_csv(path_sample_submission, dtype={'fullVisitorId': 'str'})
    except IOError:
        print(str(path_sample_submission) + " is not found.")
    assert sample_submission.shape[0] == 617242, \
        "test_ids contains {} rows, while 617,242 are expected.".format(sample_submission.shape[0])

    print("Fitting model...")
    model = predictor_cls(**params)
    model.fit(train_x, train_y)

    print("Predicting...")
    prediction = pd.DataFrame()
    prediction["fullVisitorId"] = test_x["fullVisitorId"]
    prediction["PredictedLogRevenue"] = model.predict(test_x)

    print("Creating submission...")
    submission = pd.merge(sample_submission[["fullVisitorId"]], prediction, on="fullVisitorId", how="left")
    submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].fillna(0)

    if save:
        print("Saving submission file...")
        try:
            submission.to_csv(path, index=False)
        except IOError:
            print("Path not found, saving in working directory instead under name 'submission.csv'.")
            submission.to_csv("submission.csv", index=False)

    return submission
