# Pneumonia
Creation of pneumonia determination model

A.
The original (pneumonia.py) was taken from the sample code "Pneumonia Classification on TPU" on the Keras homepage https://keras.io/examples/vision/xray_classification_with_tpus/.

As a change
1. Delete TPU related items
2. Changed the input image from 180 x 180 to 360 x 360 (this was slightly more accurate) and accordingly added a convolution layer (2 separable convolution operations, batch normalization, pooling) and a dropout layer
     x = conv_block(512, x)
     x = layers.Dropout(0.4)(x)
3. Save the best model as "xray_model_original.h5"
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("xray_model_original.h5", save_best_only=True)
 


B.
Fine-tuning used image data collected in another project. In order to support domain shift (https://qiita.com/kaco/items/f186af6abad626ce0374), Domain Adaptation (https://www.mi.t.u-tokyo.ac.jp/default/domain_adaptation) is required. It is for the purpose of doing.
As a change from the above model
1. Arrange the input X-rays as shown below and read them using tf.keras.utils.image_dataset_from_directory (https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory)
finetuningXP/
...normalXP/
......normal_image_1.jpg
......normal_image_2.jpg
......
...pneumoniaXP /
...... pneumonia_image_1.jpg
...pneumonia_image_2.jpg
......
ds = tf.keras.utils.image_dataset_from_directory(
     os.path.join(".", finetuningXP_dir),
     label_mode = 'int',
     color_mode = 'rgb',
     # color_mode='grayscale',
     validation_split = 0.1,
     # interpolation='lanczos3',
     shuffle=True,
     subset = "both",
     seed = 12345,
     image_size = (IMAGE_SIZE[0], IMAGE_SIZE[1]),
     batch_size = BATCH_SIZE,
)
2. Change the learning rate from 0.015 to 0.001 and set the maximum learning number to 30.
initial_learning_rate = 0.001
epochs=30
3. Save the best model as "xray_model_finetuning.h5"
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("xray_model_finetuning.h5", save_best_only=True)
 


C.
The correct response rate of the two models mentioned above was evaluated using the corona pneumonia X-rays of this study (evaluate.py). As a result, the correct answer rate improved from 64% in the original model to 74% after fine-tuning.
normal TP/Total: 77 / 115
pneumonia TP/Total: 151 / 193
accuracy: 0.7402597402597403
recall for normal: 0.6695652173913044
recall for pneumonia: 0.7823834196891192
precision for normal: 0.6470588235294118
precision for pneumonia: 0.798941798941799
Found 308 files belonging to 2 classes.
datasets <BatchDataset element_spec=(TensorSpec(shape=(None, 360, 360, 3), dtype=tf.float32, name=None), TensorSpec(shape=(None,), dtype=tf.int32, name=None) )>
['normalXP', 'pneumoniaXP']
13/13 [==============================] - 2s 60ms/step - loss: 1.1742 - binary_accuracy: 0.6429 - precision : 0.6397 - recall: 0.9845
original model: {'loss': 1.1742496490478516, 'binary_accuracy': 0.6428571343421936, 'precision': 0.6397306323051453, 'recall': 0.984455943107605}
13/13 [==============================] - 2s 61ms/step - loss: 0.6283 - binary_accuracy: 0.7370 - precision : 0.7979 - recall: 0.7772
fine-tuned model: {'loss': 0.6283097863197327, 'binary_accuracy': 0.7370129823684692, 'precision': 0.7978723645210266, 'recall': 0.7772020697593689}


D.
Convert keras (python) model to Javascript model (keras_js_converter.py) [https://www.tensorflow.org/js/tutorials/conversion/import_keras?hl=ja. It doesn't work as is (https://github.com/tensorflow/tfjs/issues/1739), so in model.json, change "kernel_initializer" to "depthwise_initializer", "kernel_regularizer" to "depthwise_intializer", and "kernel_contraint" to " depthwise_constraint".
