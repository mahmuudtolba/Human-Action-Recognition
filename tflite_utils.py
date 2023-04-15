import tensorflow as tf
from tensorflow import keras
TF_LITE_FILE_NAME = 'action.tflite'
model = keras.models.load_model('action.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
open(TF_LITE_FILE_NAME , 'wb').write(tflite_model)