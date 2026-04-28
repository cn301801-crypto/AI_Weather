def combine(img_pred, num_pred):
    if img_pred is None:
        return num_pred
    if img_pred == num_pred:
        return img_pred
    return num_pred