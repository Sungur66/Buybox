from tensorflow.keras.layers import DepthwiseConv2D
def custom_depthwise_conv(config):
  # 'groups' argümanını kaldırın (desteklenmediğini varsayalım)
  new_config = {k: v for k, v in config.items() if k != 'groups'}
  return DepthwiseConv2D(**new_config)
# Model yüklerken (TensorFlow/Keras örneği)
from tensorflow.keras.models import load_model
custom_objects = {'custom_depthwise_conv': custom_depthwise_conv}
model = load_model('trained_model8480.h5', custom_objects=custom_objects)