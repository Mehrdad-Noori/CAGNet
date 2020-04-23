import tensorflow.keras.backend as K


def custom_loss(y_true, y_pred, alpha1=1, alpha2=0.5, alpha3=1):
    """
    The designed loss function consists of precision, recall and mean squared error as follows:
        loss = a1 * precision_loss + a2 * recall_loss + a3 * mae_loss
    where α 1 , α 2 and α3 are the balance parameters.
    """
    prec, recall = precision_recall(y_true, y_pred)
    mae = mean_squared_error(y_true, y_pred)

    loss = alpha1 * (1 - prec) + alpha2 * (1 - recall) + alpha3 * mae
    return loss


def precision_recall(y_true, y_pred):
    eps = K.constant(1e-22, dtype='float32')

    y_true = K.cast(K.argmax(y_true, axis=-1), K.floatx())
    y_true = K.expand_dims(y_true, axis=-1)
    y_pred = y_pred[..., 1:2]

    y_true = K.batch_flatten(y_true)
    y_pred = K.batch_flatten(y_pred)

    sum_pt = K.sum(y_pred * y_true, axis=-1)
    sum_p = K.sum(y_pred, axis=-1)
    sum_t = K.sum(y_true, axis=-1)

    prec = sum_pt / (sum_p + eps)
    recall = sum_pt / (sum_t + eps)

    return K.mean(prec), K.mean(recall)


def mean_squared_error(y_true, y_pred):
    y_true = K.cast(K.argmax(y_true, axis=-1), K.floatx())
    y_true = K.expand_dims(y_true, axis=-1)
    y_pred = y_pred[..., 1:2]
    return K.mean(K.abs(y_pred - y_true), axis=-1)
