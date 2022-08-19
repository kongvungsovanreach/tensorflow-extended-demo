from tfx import v1 as tfx
import tensorflow as tf
import tensorflow_transform as tft
from typing import Dict, List, Text
from tensorflow_transform import TFTransformOutput
from tfx_bsl.public import tfxio
from absl import logging

BATCH_SIZE=64
label='Survived'

def input_func(file_pattern: List[Text],
              data_accessor: tfx.components.DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
    
    return data_accessor.tf_dataset_factory(
      file_pattern,
      tfxio.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=label),
      tf_transform_output.transformed_metadata.schema)

def get_model(tf_transform_output: TFTransformOutput, show_summary=False) -> tf.keras.Model:
    feature_spec = tf_transform_output.transformed_feature_spec().copy()
    feature_spec.pop(label)

    inputs = {}
    for key, spec in feature_spec.items():
        if isinstance(spec, tf.io.VarLenFeature):
            inputs[key] = tf.keras.layers.Input(
              shape=[None], name=key, dtype=spec.dtype, sparse=True)
        elif isinstance(spec, tf.io.FixedLenFeature):
          # TODO(b/208879020): Move into schema such that spec.shape is [1] and not
          # [] for scalars.
            inputs[key] = tf.keras.layers.Input(
              shape=spec.shape or [1], name=key, dtype=spec.dtype)
        else:
            raise ValueError('Spec type is not supported: ', key, spec)

    output = tf.keras.layers.Concatenate()(tf.nest.flatten(inputs))
    output = tf.keras.layers.Dense(100, activation='relu')(output)
    output = tf.keras.layers.Dense(70, activation='relu')(output)
    output = tf.keras.layers.Dense(50, activation='relu')(output)
    output = tf.keras.layers.Dense(20, activation='relu')(output)
    output = tf.keras.layers.Dense(1, activation="sigmoid")(output)
    
    model = tf.keras.Model(inputs=inputs, outputs=output)
    
    model.compile(
      # loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
      # optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
      # metrics=[tf.keras.metrics.BinaryAccuracy()]
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.TruePositives(),
        ],
    )
    if show_summary:
        model.summary()
    
    return model

def get_example_sig(model, tf_transform_output):
    model.tft_layer_inference = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_example):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_feature_spec.pop(label)
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_inference(raw_features)
        logging.info('serve_transformed_features = %s', transformed_features)

        outputs = model(transformed_features)
        return {'outputs': outputs}

    return serve_tf_examples_fn


def get_transform_feature_sig(model, tf_transform_output):
    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_eval(raw_features)
        logging.info('eval_transformed_features = %s', transformed_features)
        return transformed_features

    return transform_features_fn


def export_serving_model(tf_transform_output, model, output_dir):
    model.tft_layer = tf_transform_output.transform_features_layer()

    signatures = {
      'serving_default':
          get_example_sig(model, tf_transform_output),
      'transform_features':
          get_transform_feature_sig(model, tf_transform_output),
    }
    
    print(output_dir)
    model.save(output_dir, save_format='tf', signatures=signatures)

def run_fn(fn_args: tfx.components.FnArgs):
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
    train_dataset = input_func(fn_args.train_files, fn_args.data_accessor, 
                            tf_transform_output, BATCH_SIZE)
    eval_dataset = input_func(fn_args.eval_files, fn_args.data_accessor, 
                           tf_transform_output, BATCH_SIZE)
    
    model = get_model(tf_transform_output, True)
    
    
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='batch')
    
    model.fit(
      train_dataset,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])
    
    print(model)
    export_serving_model(tf_transform_output, model, fn_args.serving_model_dir)