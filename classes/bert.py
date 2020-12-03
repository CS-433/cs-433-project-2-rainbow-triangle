from classes.abstract_model import AbstractModel
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures, AdamWeightDecay, WarmUp
import tensorflow as tf


class Bert(AbstractModel):

  def __init__(self, weights_path):
    """
    Instantiates the model for Bert classification.

    :param weights_path: specifies where load/store weights of the model
    """
    super().__init__(weights_path)

    self.__model = TFBertForSequenceClassification.from_pretrained(
      'bert-base-uncased')
    self.__tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

  def fit_predict(self, X, Y, ids_test, X_test, prediction_path, batch_size=24,
                  epochs=1):
    """
    Fits the model and performs the prediction.

    :param X: data
    :param Y: labels
    :param batch_size: specifies how many datapoints to use for each step
    """

    # Converting the tweets to have a good input for BERT
    train_input_examples, validation_input_examples = \
      self.__convert_data_to_examples(X=X, Y=Y, split_size=0.1)

    train_data_size = len(train_input_examples)

    train_data = self.__convert_examples_to_tf_dataset(
      list(train_input_examples))
    train_data = train_data.shuffle(100).batch(batch_size).repeat(2)

    validation_data = self.__convert_examples_to_tf_dataset(
      list(validation_input_examples))
    validation_data = validation_data.batch(batch_size)

    steps_per_epoch = int(train_data_size / batch_size)
    num_train_steps = steps_per_epoch * epochs

    decay_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
      initial_learning_rate=2e-5,
      decay_steps=num_train_steps,
      end_learning_rate=0)

    warmup_schedule = WarmUp(
      initial_learning_rate=2e-5,
      decay_schedule_fn=decay_schedule,
      warmup_steps=(num_train_steps * 0.1))

    optimizer = AdamWeightDecay(learning_rate=warmup_schedule,
                                epsilon=1e-08,
                                clipnorm=1.0)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    self.__model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

    print('Fitting the model...')

    self.__model.fit(train_data, epochs=epochs, validation_data=validation_data)

    print('Saving the weights...')
    self.__model.save_pretrained(f'{self._weights_path}model')

    print('Predicting...')
    self.predict(ids_test, X_test, prediction_path)

  def predict(self, ids, X, path, from_weights=False):
    """
    Performs the predictions.

    :param ids: ids of new data
    :param X: new data to predict
    :param path: specifies where to store the submission file
    """

    if from_weights:
      self.__model = TFBertForSequenceClassification.from_pretrained(
        f'{self._weights_path}model')

    predictions = []

    for i, tweet in enumerate(X):
      feature = self.__tokenizer.encode_plus(text=tweet, return_tensors='tf')
      output = self.__model(feature)[0].numpy().squeeze().argmax()
      predictions.append(output)

      if i % 100 == 0:
        print(f'Step: {i}')

    AbstractModel._create_submission(ids, predictions, path)

  def __convert_examples_to_tf_dataset(self, data, max_length=128):
    """
    Performs the tokenization where each word of each document has a max_length.

    :param data: input data
    :param max_length: length of the tokenization
    :return: precessed data
    """
    features = []  # -> will hold InputFeatures to be converted later

    for e in data:
      # Documentation is really strong for this method, so please take a look
      # at it
      input_dict = self.__tokenizer(
        e.text_a,
        add_special_tokens=True,
        max_length=max_length,  # truncates if len(s) > max_length
        return_token_type_ids=False,
        return_attention_mask=True,
        padding='max_length',  # pads to the right by default
        truncation=True
      )

      input_ids, attention_mask = (
        input_dict['input_ids'],
        input_dict['attention_mask'])

      features.append(
        InputFeatures(
          input_ids=input_ids, attention_mask=attention_mask,
          label=e.label
        )
      )

    def gen():
      for f in features:
        yield (
          {
            'input_ids': f.input_ids,
            'attention_mask': f.attention_mask,
          },
          f.label,
        )

    return tf.data.Dataset.from_generator(
      gen,
      (
        {
          'input_ids': tf.int32,
          'attention_mask': tf.int32,
        },
        tf.int64
      ),
      (
        {
          'input_ids': tf.TensorShape([None]),
          'attention_mask': tf.TensorShape([None]),
        },
        tf.TensorShape([]),
      ),
    )

  @staticmethod
  def __convert_data_to_examples(X, Y, split_size=0.2):
    """
    Function to transform the data in a format suitable for BERT.

    :param X: input data
    :param Y: input labels
    :param split_size: specifies the ratio to split data in train/test
    :return: transformed data
    """
    X_train, X_test, Y_train, Y_test = AbstractModel._split_data(
      X=X,
      Y=Y,
      split_size=split_size)

    train_input_examples = []

    for text, label in zip(X_train, Y_train):
      train_input_examples.append(
        InputExample(guid=None, text_a=text, text_b=None, label=label))

    validation_input_examples = []

    for text, label in zip(X_test, Y_test):
      validation_input_examples.append(
        InputExample(guid=None, text_a=text, text_b=None, label=label))

    return train_input_examples, validation_input_examples
