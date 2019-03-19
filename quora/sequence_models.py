import numpy as np

from keras.layers import Input, Embedding, SpatialDropout1D, Bidirectional, LSTM, Flatten, Concatenate, Dense
from keras.initializers import glorot_normal, orthogonal
from keras.models import Model

from sklearn.model_selection import StratifiedKFold
from sklearn import utils

from common.nn.elements import Capsule, DropConnect, Attention


def classify_based_on_probs(predictions, boundary=0.2):
    """Classifies a prediction as 0 or 1 based on the given probability

    Parameters
    ----------
    predictions: np.array
        The array of predictions. Each prediction is a float on range [0, 1]
    boundary: float
        The boundary of the classification. When a prediction is above this boundary
        it is classified as 1 else as 0.

    Returns
    -------
    An np.array in which each instance is classified as 0 or 1.

    """
    labels = (predictions > boundary)
    return np.array(labels, dtype=int)


def evaluate_classification_boundary(predictions, true_labels, boundary, evaluation_func):
    """Function that calculates the score based on a given boundary and evaluation function.

    Parameters
    ----------
    predictions: np.array
        An array with predictions. Each prediction is a float on range [0, 1]
    true_labels: np.array
        The real binary y values.
    boundary: float
        The boundary of the classification. When a prediction is above this boundary
        it is classified as 1 else as 0.
    evaluation_func: function
        The evaluation function that should decide on the score obtained with the given boundary

    Returns
    -------
    score: fleat
        The score related to the given boundary

    """
    predicted_labels = classify_based_on_probs(predictions, boundary=boundary)
    score = evaluation_func(predicted_labels, true_labels)
    return score


def get_best_threshold(val_predictions, val_true, low=0.1, high=0.8, steps=0.01):
    """Function that defines the best threshold for classification.

    Parameters
    ----------
    val_predictions: np.array
        The array with predictions of the validation set. Each prediction is a float on range [0, 1]
    val_true: np.array
        The binary y values of the validation set.
    low: float, default 0.1
        The lowest threshold tested
    high: float, default 0.8
        The highest threshold tested
    steps: float, default 0.01
        The step size with which all thresholds between low and high are evaluated

    Returns
    -------
    best_b: float
        The threshold that gave the best score
    best_score: fleat
        The score related to the best threshold

    """
    best_b, best_score = 0, 0
    for b in np.arange(low, high, steps):
        score = evaluate_classification_boundary(val_predictions, val_true, b, binary_f1score)
        if score > best_score:
            best_b = b
            best_score = score

    print("Best F1 score is {} at threshold {}".format(best_score, best_b))
    return best_b, best_score


def binary_f1score(y_predict, y_true):
    """F1 score for boolean / binary problems.

    Parameters
    ----------
    y_predict: np.array
        The predicted binary y values.
    y_true: np.array
        The true binary y values.

    Returns
    -------
    f: float
        The F1 score of the prediction.

    """
    true_positives = np.sum((y_predict == 1) & (y_true == 1))
    false_positives = np.sum((y_predict == 1) & (y_true == 0))
    false_negatives = np.sum((y_predict == 0) & (y_true == 1))
    recall = true_positives / (true_positives + false_negatives)
    precision = true_positives / (true_positives + false_positives)
    f = 2 * (precision * recall) / (precision + recall)
    return f


def create_sequence_model(embedding_matrix, word_index, max_words):
    """This function creates a deep learning model for the Quora competition.

    Parameters
    ----------
    embedding_matrix: np.array
        The matrix of embeddings used in the embedding layer
    word_index: dict
        The word_index obtained by the tokenizer
    max_words: int
        The maximum number of words in a sequence

    Returns
    -------
    A deep learning model
    """
    input_layer = Input(shape=(75, ), name='input_layer')
    x = Embedding(len(word_index) + 1, embedding_matrix.shape[1], weights=[embedding_matrix], trainable=False, input_shape=(max_words,))(input_layer)
    x = SpatialDropout1D(rate=0.24)(x)
    x = Bidirectional(layer=LSTM(80, return_sequences=True, kernel_initializer=glorot_normal(seed=1029),
                                 recurrent_initializer=orthogonal(gain=1.0, seed=1029)), name='bidirectional_lstm')(x)

    # Capsule layer
    capsule = Capsule(num_capsule=10, dim_capsule=10, routings=4, share_weights=True)(x) # noqa
    capsule = Flatten()(capsule)
    capsule = DropConnect(Dense(32, activation="relu"), prob=0.01)(capsule)

    # Attention layer
    atten = Attention(step_dim=75, name='attention')(x)
    atten = DropConnect(Dense(16, activation="relu"), prob=0.05)(atten)

    # Concatenate Capsule and Attention layer
    x = Concatenate(axis=-1)([capsule, atten])

    output_layer = Dense(units=1, activation='sigmoid', name='output')(x)
    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam')

    return model


def cross_validate_and_predict(x, y, xtest, embedding_matrix, word_index, max_words, folds=4):
    """This function applies cross validation on the dataset. It predicts the y values of
    the test sets based on these folds.

    Parameters
    ----------
    x: np.array
        The x-values of the training set
    y: np.array
        The y-values of the training set
    xtest: np.array
        The x-values of the test set
    embedding_matrix: np.array
        The matrix of embeddings used in the embedding layer
    word_index: dict
        The word_index obtained by the tokenizer
    max_words: int
        The maximum number of words in a sequence
    folds: int, default 4
        The number of folds used in cross-validation

    Returns
    -------
    predictions: np.array
        The binary predictions of the test set

    """
    print("Start cross validation..")
    kfold = StratifiedKFold(n_splits=folds, random_state=99, shuffle=True)
    thresholds, scores = [], []
    probas = np.zeros(len(xtest))
    for train_index, val_index in kfold.split(x, y):
        # Start fold
        x_train, x_val, y_train, y_val = x[train_index], x[val_index], y[train_index], y[val_index]

        # Create model
        model = create_sequence_model(embedding_matrix, word_index, max_words)

        # Get weights
        weights = utils.class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)

        # Fit model
        model.fit(x_train, y_train, batch_size=512, epochs=6,
                  validation_data=(x_val, y_val), verbose=2, class_weight=weights)

        # Predict validation labels
        predicted_y_val = model.predict([x_val], batch_size=1024, verbose=2).flatten()

        # Get and save best threshold and score
        boundary, f1score = get_best_threshold(predicted_y_val, y_val)
        thresholds.append(boundary)
        scores.append(f1score)
        print("Fold completed. Validation score = {} at threshold {}".format(f1score, boundary))

        # Predict test set with the model
        probas += (model.predict([xtest], batch_size=1024, verbose=2).flatten() / folds)

    # Get predictions based on all folds
    predictions = classify_based_on_probs(probas, boundary=np.mean(thresholds))

    return predictions
