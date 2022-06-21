import os
import joblib
import tensorflow as tf
import Models.joy_and_anger.joy_and_anger_utils as joy_and_anger_lib_utils
import app

# export our trained vocab list since this wasn't done in the training process
def export_vocab_list(data_fp):
    """
    Creates a vocab txt file from train.txt and saves it in the same folder as joy_and_anger_vocab.txt
    :param data_fp: full path for train.txt
    :return:
    """
    res = []
    with open(data_fp, "r+") as f:
        lines = f.readlines()
        for line in lines:
            # if the line has multiple semi-colons it will break.
            vocab_line, label = line.split(";")
            label = label.strip()
            if label == "joy" or label == "anger":
                res.append(vocab_line)
    target_fp = os.path.join(os.path.dirname(data_fp), "joy_and_anger_vocab.txt")
    with open(target_fp, "w+") as f:
        f.write("\n".join(res))

def load_saved_model_locally(model_fn):
    """
    Loads a deployed model into memory
    :param model_fn: name of the model
    :return: model with adapted TextVectorization and loaded weights
    """
    model = joy_and_anger_lib_utils.create_model(None, os.path.join(app.root(), "datasets", "emotions", "joy_and_anger_vocab.txt"))
    model.load_weights(os.path.join(app.root(), "Models", model_fn, model_fn))

    return model


# pickle is not compatible with the preprocessing of TextVectorization
# pickle is also discouraged from usage in production code.
# pickle reference is kept for future implementations if it will become necessary

# def pickle_model(tf_model_name:str, pickle_model_name:str):
#     '''
#     Loads a tf SavedModel model and pickles the object.
#     input:
#     tf_model_name: tf model name in string in ../training/models/*tf_model_name*
#     pickle_model_name: output pickled model name in string in ../Models/pickled/*pickle_model_name*
#     '''
#     tf_model_path = os.path.join(os.getcwd(), "..", "training", "models", tf_model_name)
#
#     model = tf.keras.models.load_model(tf_model_path)
#     dest = os.path.join(os.getcwd(), "..", "Models", "pickled", pickle_model_name)
#     with open(dest, 'wb') as f:
#         joblib.dump(model, f)
#
# def test_model(model_fn, input):
#     with open(os.path.join(os.getcwd(), "..", "Models", "pickled", model_fn), "rb") as f:
#         model = joblib.load(f)
#         print(model.predict(input))
