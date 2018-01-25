from data_utils import load_tl_extracts
#import numpy as np
from sklearn.model_selection import train_test_split
from tl_classifier_cnn import TLClassifierCNN, TLLabelConverter


# load data
desired_dim = (32,32)
#data_dirs = ['data/tl-extract-train', 'data/tl-extract-additional']
data_dirs = ['data/tl-extract-train', 'data/tl-extrac-additional-train']
x, y = load_tl_extracts(data_dirs, desired_dim)
# x is image in OpenCV imread format. pixels are uint8 from 0 to 255. shape is H, W, C. C is ordered BGR
# y here are strings like 'green' etc


# filter data with only labels relevant for us
converter = TLLabelConverter()
x, y = converter.filter(x, y)


# split into train/test
pct_train = 85.
pct_valid = 15.
random_state = 123
train_features, val_features, train_labels, val_labels = \
  train_test_split(x, y, train_size = pct_train/100., test_size = pct_valid/100., random_state = random_state)


# define classifier
tlc = TLClassifierCNN()
checkpoint_dir = 'ckpt/model.ckpt'
tlc.restore_checkpoint(checkpoint_dir)

features_shape = ((None,) + train_features.shape[1:])
labels_shape = ((None,) + converter.get_shape())
#save_file = 'ckpt/model.ckpt'
summary_dir = 'train_summaries'
learning_rate = 0.0001
tlc = TLClassifierCNN(learning_rate=learning_rate)

epochs = 50
batch_size = 250
max_iterations_without_improvement = 40
dropout_keep_probability=0.7
#summary_dir = 'summaries'

best_validation_accuracy = \
    tlc.train(train_images             = train_features,
              train_labels_str         = train_labels,
              validation_images        = val_features,
              validation_labels_str    = val_labels,
              dropout_keep_probability = dropout_keep_probability,
              batch_size               = batch_size,
              epochs                   = epochs,
              max_iterations_without_improvement = max_iterations_without_improvement,
              checkpoint_dir           = checkpoint_dir,
              summary_dir              = summary_dir)

# save trained model
model_dir = 'model'
tlc.save_model(model_dir)
tlc.close_session()

# tensorboard --logdir=./summaries


