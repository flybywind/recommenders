# -*- coding:utf-8 -*-
"""
Author:
    Weiyu Cheng, weiyu_cheng@sjtu.edu.cn

Reference:
    [1] Cheng, W., Shen, Y. and Huang, L. 2020. Adaptive Factorization Network: Learning Adaptive-Order Feature
         Interactions. Proceedings of the AAAI Conference on Artificial Intelligence. 34, 04 (Apr. 2020), 3609-3616.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

class LogTransformLayer(keras.layers.Layer):
    """Logarithmic Transformation Layer in Adaptive factorization network, which models arbitrary-order cross features.

      Input shape
        - 3D tensor with shape: ``(batch_size, field_size, embedding_size)``.
      Output shape
        - 2D tensor with shape: ``(batch_size, ltl_hidden_size*embedding_size)``.
      Arguments
        - **field_size** : positive integer, number of feature groups
        - **embedding_size** : positive integer, embedding size of sparse features
        - **ltl_hidden_size** : integer, the number of logarithmic neurons in AFN
      References
        - Cheng, W., Shen, Y. and Huang, L. 2020. Adaptive Factorization Network: Learning Adaptive-Order Feature
         Interactions. Proceedings of the AAAI Conference on Artificial Intelligence. 34, 04 (Apr. 2020), 3609-3616.
    """

    def __init__(self, field_size, embedding_size, ltl_hidden_size, l2_reg=None):
        super(LogTransformLayer, self).__init__()
        self.field_size = field_size
        self.embedding_size = embedding_size
        self.ltl_hidden_size = ltl_hidden_size
        self.l2_regularizer = None if l2_reg is None else keras.regularizers.L2(l2_reg)
        self.bn = [keras.layers.BatchNormalization() for _ in range(2)]
        self._build_params()

    def _build_params(self):
        self.ltl_weights = self.add_weight(name='ltl_weights',
                                           shape=(self.field_size, self.ltl_hidden_size),
                                           initializer=tf.initializers.truncated_normal(mean=0.0, stddev=0.1),
                                           regularizer=self.l2_regularizer,
                                           dtype=tf.float32)

        self.ltl_biases = self.add_weight(name='ltl_bias',
                                          shape=(1, 1, self.ltl_hidden_size),
                                          initializer=tf.initializers.zeros(),
                                          dtype=tf.float32)

    def call(self, inputs, training):
        # Avoid numeric overflow
        afn_input = tf.clip_by_value(tf.abs(inputs), clip_value_min=1e-7, clip_value_max=float("Inf"))
        # Transpose to shape: ``(batch_size,embedding_size,field_size)``
        afn_input_trans = tf.transpose(afn_input, perm=[0, 2, 1])
        # Logarithmic transformation layer
        ltl_result = tf.math.log(afn_input_trans)
        ltl_result = self.bn[0](ltl_result, training)
        ltl_result = tf.matmul(ltl_result, self.ltl_weights) + self.ltl_biases
        tf.summary.histogram(name='logOrder', data=self.ltl_weights)
        ltl_result = tf.math.exp(ltl_result)
        ltl_result = self.bn[1](ltl_result, training)
        out_shape = tf.shape(ltl_result)
        ltl_result = tf.reshape(ltl_result, [-1, out_shape[1]*out_shape[2]], name='flatten')
        return ltl_result


class DenseDropBatchNorm(keras.layers.Dense):

    def __init__(self, units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer="glorot_uniform",
                 kernel_regularizer=None,
                 dropout=0.0,
                 use_bn=False, **kwargs):

        super(DenseDropBatchNorm, self).__init__(units, activation, use_bias, kernel_initializer,
                                                 kernel_regularizer=kernel_regularizer,
                                                 dtype=tf.float32, **kwargs)
        self.batch_norm = None
        self.dropout = None
        if use_bn:
            self.batch_norm = keras.layers.BatchNormalization()
        if dropout > 0:
            self.dropout = keras.layers.Dropout(dropout)

    def call(self, inputs, training):
        y = super(DenseDropBatchNorm, self).call(inputs)
        if self.batch_norm:
            y = self.batch_norm(y, training)
        if self.dropout:
            y = self.dropout(y, training)
        return y


class AFN(keras.Model):
    """Instantiates the Adaptive Factorization Network architecture.

    In DeepCTR-Torch, we only provide the non-ensembled version of AFN for the consistency of model interfaces. For the ensembled version of AFN+, please refer to https://github.com/WeiyuCheng/DeepCTR-Torch (Pytorch Version) or https://github.com/WeiyuCheng/AFN-AAAI-20 (Tensorflow Version).

    :param field_cnt: int, count of fields
    :param feature_cnt: int, count of features of all fields
    :param ltl_hidden_size: integer, the number of logarithmic neurons in AFN
    :param afn_dnn_hidden_units: list, list of positive integer or empty list, the layer number and units in each layer of DNN layers in AFN
    :param l2_reg: float. L2 regularizer strength applied to all weights
    :param init_std: float,to use as the initialize std of embedding vector
    :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
    :param dnn_activation: Activation function to use in DNN
    :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
    :param device: str, ``"cpu"`` or ``"cuda:0"``
    :param gpus: list of int or torch.device for multiple gpus. If None, run on `device`. `gpus[0]` should be the same gpu with `device`.
    :return: A PyTorch model instance.

    """

    def __init__(self, field_cnt, feature_cnt, **kwargs):
        super(AFN, self).__init__()

        self.field_cnt = field_cnt
        self.feature_cnt = feature_cnt
        self.embedding_size = 128
        self.ltl_hidden_size = 256
        self.afn_dnn_hidden_units = (256, 128)
        self.ensamble_dnn_units = ()
        self.l2_reg = 0.00001
        self.init_std = 0.0001
        self.dnn_dropout = 0.0
        self.dnn_activation = 'relu'
        self.output_activation = 'elu'
        self.use_bn = False
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.ltl = LogTransformLayer(self.field_cnt, self.embedding_size, self.ltl_hidden_size,
                                     l2_reg=self.l2_reg)
        self.afn_dnn_list = [DenseDropBatchNorm(out_unit,
                                                activation=self.dnn_activation,
                                                kernel_regularizer=keras.regularizers.L2(self.l2_reg),
                                                kernel_initializer=tf.initializers.he_uniform(),
                                                dropout=self.dnn_dropout, use_bn=self.use_bn)
                             for out_unit in self.afn_dnn_hidden_units]

        self.ensamble_dnn = [DenseDropBatchNorm(out_unit,
                                                activation=self.dnn_activation,
                                                kernel_regularizer=keras.regularizers.L2(self.l2_reg),
                                                kernel_initializer=tf.initializers.he_uniform(),
                                                dropout=self.dnn_dropout, use_bn=self.use_bn)
                             for out_unit in self.ensamble_dnn_units]
        if len(self.ensamble_dnn_units) > 0:
            self.ensamble_dnn.append(
                keras.layers.Dense(1, activation=self.output_activation, kernel_regularizer=keras.regularizers.L2(self.l2_reg)))
        self.output_layer = keras.layers.Dense(1, activation=self.output_activation, kernel_regularizer=keras.regularizers.L2(self.l2_reg))
        self._build_params()

    def _build_params(self):
        # self.ltl.build(input_shape)
        # for layer in self.afn_dnn_list:
        #     layer.build(input_shape)
        # self.output_layer.
        self.sparse_linear_weight = self.add_weight(name='sparse_weight', shape=(self.feature_cnt, 1),
                                                    initializer=tf.initializers.glorot_normal(),
                                                    regularizer=keras.regularizers.L2(self.l2_reg),
                                                    dtype=tf.float32)
        self.sparse_linear_b = self.add_weight(name='sparse_linear_b', shape=(1,),
                                               initializer='zero',
                                               regularizer=keras.regularizers.L2(self.l2_reg),
                                               dtype=tf.float32)
        self.embedding = self.add_weight(name='emb_mat', shape=(self.feature_cnt, self.embedding_size),
                                         initializer=tf.initializers.glorot_normal(),
                                         regularizer=keras.regularizers.L2(self.l2_reg),
                                         dtype=tf.float32)

    def _build_linear(self, inputs):
        # read indices of one-hot sparse encoding
        indices = inputs['oh_indices']
        values = inputs['oh_values']
        dense_shape = inputs['oh_shape']
        x = tf.SparseTensor(indices, values, dense_shape)
        tf.summary.histogram("linear/w", self.sparse_linear_weight)
        tf.summary.histogram("linear/b", self.sparse_linear_b)
        return tf.add(tf.sparse.sparse_dense_matmul(x, self.sparse_linear_weight), self.sparse_linear_b)

    def _build_embedding(self, inputs):
        """The field embedding layer. MLP requires fixed-length vectors as input.
        This function makes sum pooling of feature embeddings for each field.
        Args:
            inputs tf.Tensors: dict of tensors
        Returns:
            embedding:  The result of field embedding layer, with size of #_fields * #_dim.
            embedding_size: #_fields * #_dim
        """
        field_indices = inputs['indices']  # indices in sparse mat of [batch x field]
        field_values = inputs['values']  # values in the sparse mat, representing the feature id
        field_weights = inputs['weights']
        field_shape_batch = inputs['shape']
        fm_sparse_index = tf.SparseTensor(field_indices, field_values, field_shape_batch)
        # for category features, the weight = 1
        # for float value features, the weight = float value, eg, history ctr
        fm_sparse_weight = tf.SparseTensor(field_indices, field_weights, field_shape_batch)
        embedding = tf.nn.embedding_lookup_sparse(
            params=self.embedding,
            sp_ids=fm_sparse_index,
            sp_weights=fm_sparse_weight,
            combiner="sum",
        )
        embedding = tf.reshape(
            embedding, [-1, self.field_cnt, self.embedding_size]
        )
        return embedding

    def _build_dnn(self, inputs, training):
        x = inputs
        for layer in self.afn_dnn_list:
            x = layer(x, training)
        return x

    def call(self, inputs, training):
        logit = self._build_linear(inputs)
        afn_input = self._build_embedding(inputs)
        ltl_result = self.ltl(afn_input, training)
        afn_logit = self._build_dnn(ltl_result, training)
        logit += afn_logit
        aft_logit = self.output_layer(afn_logit)

        if len(self.ensamble_dnn) > 0:
            x = tf.reshape(afn_input, [-1, self.embedding_size*self.field_cnt])
            for layer in self.ensamble_dnn[:-1]:
                x = layer(x, training)
            x = self.ensamble_dnn[-1](x)
            aft_logit += x

        return tf.squeeze(aft_logit)

    def predict(self, inputs):
        pred_logit = self(inputs, False)
        return tf.nn.sigmoid(pred_logit)

    def loss_function(self, pred, targets):
        pred_loss = tf.reduce_mean(keras.losses.binary_crossentropy(targets, pred, from_logits=True))
        reg_loss = tf.reduce_sum(self.losses)
        tf.summary.scalar(name='entropy_loss', data=pred_loss)
        tf.summary.scalar(name='regularized_loss', data=reg_loss)
        return pred_loss + reg_loss

    def metrics(self, pred_list, target_list):
        if not hasattr(self, "metrics_dict"):
            self.metrics_dict = {'AUC': tf.metrics.AUC(), 'MAE': tf.metrics.MeanAbsoluteError()}

        out = {}
        for pred, target in zip(pred_list, target_list):
            for m, calc in self.metrics_dict.items():
                calc.update_state(y_pred=pred, y_true=target)

        for m, calc in self.metrics_dict.items():
            out[m] = calc.result()
            tf.summary.scalar(f"valid_{m}", out[m])
            calc.reset_state()
        return out

    def train_model(
            self, logdir,
            train_data,
            start_learning_rate=0.01,
            end_learning_rate=1e-5,
            decay_power=0.5,
            validation_data=None,
            batch_size=None,
            epochs=1,
            verbose="auto",
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
    ):
        from recommenders.models.deeprec.io.afn_iterator import AFNFFMTextIterator
        train_iterator = AFNFFMTextIterator(batch_size, self.feature_cnt, self.field_cnt)
        # train_iterator = iter(dataset_iterator.load_data_from_file(train_data))
        valid_iterator = AFNFFMTextIterator(validation_batch_size, self.feature_cnt, self.field_cnt)
        # valid_iterator = iter(dataset_iterator.load_data_from_file(validation_data))

        lr_schedule = keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=start_learning_rate,
                                                                 end_learning_rate=end_learning_rate,
                                                                 decay_steps=100,
                                                                 cycle=True,
                                                                 power=decay_power)
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-7
        )
        optimizer.clipnorm = 5.0
        tf.summary.experimental.set_step(optimizer.iterations)
        step = 1

        train_step_signature = [
            {
                "oh_indices": tf.TensorSpec(shape=(None, 2), dtype=tf.int64),
                "oh_values": tf.TensorSpec(
                    shape=(None,), dtype=tf.float32
                ),
                "oh_shape": tf.TensorSpec(
                    shape=(2,), dtype=tf.int64
                ),
                "indices": tf.TensorSpec(
                    shape=(None, 2), dtype=tf.int64
                ),
                "values": tf.TensorSpec(
                    shape=(None, ), dtype=tf.int64
                ),
                "weights": tf.TensorSpec(
                    shape=(None, ), dtype=tf.float32
                ),
                "shape": tf.TensorSpec(
                    shape=(2,), dtype=tf.int64
                ),
                "label": tf.TensorSpec(
                    shape=(None, 1), dtype=tf.float32
                ),
            },
        ]

        tfsm_writor = tf.summary.create_file_writer(logdir=logdir)
        @tf.function(input_signature=train_step_signature)
        def train_step(data):
            target = tf.squeeze(data['label'])
            with tf.GradientTape() as tape:
                pred_logit = self(data, training=True)
                loss = self.loss_function(pred_logit, target)
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            local_step = optimizer.iterations + 1
            tf.summary.scalar("lr", lr_schedule(local_step))
            tf.summary.scalar("step", local_step)
            return loss

        with tfsm_writor.as_default():
            for e in range(1, epochs + 1):
                ts = 0
                for (inputs, impression_id_list, cnt) in tqdm(train_iterator.load_data_from_file(train_data)):
                    if step % validation_steps == 0:
                        pred_list, label_list = [], []
                        for (inputs_val, _, _) in tqdm(valid_iterator.load_data_from_file(validation_data)):
                            label_list.append(np.squeeze(inputs_val['label']))
                            pred_list.append(self.predict(inputs_val))

                        metrics_out = self.metrics(pred_list, label_list)
                        print(f"epoch: {e}, validation: {metrics_out}")

                    loss = train_step(inputs)
                    print(f"epoch: {e}, step: {step}, loss = {loss}")
                    step += 1
                    ts += 1
                    if ts >= steps_per_epoch:
                        print(f"epoch {e}, break at step {ts}")
                        break
        tfsm_writor.flush()

    def create_inputs(self, inputs_dict):
        label = inputs_dict['label']
        del inputs_dict['label']
        return inputs_dict, label