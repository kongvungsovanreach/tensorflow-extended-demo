#import modules
import os
import tensorflow as tf
import tensorflow_transform as tft

label = 'Survived'
OOV_SIZE = 10
VOCAB_SIZE = 1000
numberical_features = ['PassengerId','Age','Fare']
categorical_string_features = ['Sex', 'Cabin', 'Embarked']
categorical_numerical_features = ['Pclass', 'SibSp', 'Parch']
# bucket_features = ['Name', 'Ticket']
# feature_bucket_count = 10

def transform_feature(feature):
    return '{}_xf'.format(feature)

def fill_missing(value):
    if not isinstance(value, tf.sparse.SparseTensor):
        return value

    default = '' if value.dtype == tf.string else 0
    return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(value.indices, value.values, [value.dense_shape[0], 1]),
          default),
      axis=1)
def one_hot_encode(x, key):
    integerized = tft.compute_and_apply_vocabulary(x,
      top_k=VOCAB_SIZE,
      num_oov_buckets=OOV_SIZE,
      vocab_filename=key, name=key)
    depth = (
      tft.experimental.get_vocabulary_size_by_name(key) + OOV_SIZE)
    one_hot_encoded = tf.one_hot(
      integerized,
      depth=tf.cast(depth, tf.int32),
      on_value=1.0,
      off_value=0.0)
    return tf.reshape(one_hot_encoded, [-1, depth])

# def convert_num_to_one_hot(label_tensor: tf.Tensor, num_labels: int = 2) -> tf.Tensor:
#     one_hot_tensor = tf.one_hot(label_tensor, num_labels)
#     return tf.reshape(one_hot_tensor, [-1, num_labels])

    
def preprocessing_fn(inputs):
    outputs = {}
    
    for key in numberical_features:
        outputs[transform_feature(key)] = tft.scale_to_z_score(fill_missing(inputs[key]), name=key)
        
    for key in categorical_string_features:
        outputs[transform_feature(key)] = one_hot_encode(fill_missing(inputs[key]), key)
    
    for key in categorical_numerical_features:
        outputs[transform_feature(key)] = one_hot_encode(tf.strings.strip(
            tf.strings.as_string(fill_missing(inputs[key]))), key)
        
    outputs[label] = fill_missing(inputs[label])
        
    return outputs