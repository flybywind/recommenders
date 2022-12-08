# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import numpy as np
import pytest
import os
import os.path as osp
from recommenders.models.deeprec.deeprec_utils import (
    prepare_hparams,
    download_deeprec_resources,
)
from recommenders.models.deeprec.models.AFN import AFN

@pytest.mark.gpu
def test_afn_ffm_files(deeprec_resource_path):
    import os.path as osp
    from recommenders.models.deeprec.io.afn_iterator import AFNFFMTextIterator

    data_path = os.path.join(deeprec_resource_path, "xdeepfm")
    yaml_file = os.path.join(data_path, "xDeepFM.yaml")
    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.z20.web.core.windows.net/deeprec/",
            data_path,
            "xdeepfmresources.zip",
        )
    hparams = prepare_hparams(yaml_file, FEATURE_COUNT=1000, FIELD_COUNT=10)
    dataset = AFNFFMTextIterator(64, hparams.FEATURE_COUNT, hparams.FIELD_COUNT)
    dataset_iter = iter(dataset.load_data_from_file(osp.join(data_path, "synthetic_part_0")))
    inputs, id, cnt = next(dataset_iter)

    print(hparams, inputs, id, cnt)
    assert inputs.get('label', None) is not None
    assert inputs.get('oh_indices', None) is not None
    assert inputs.get('oh_values', None) is not None
    assert inputs.get('oh_shape', None) is not None
    assert inputs.get('indices', None) is not None
    assert inputs.get('values', None) is not None
    assert inputs.get('weights', None) is not None
    assert inputs.get('shape', None) is not None


@pytest.mark.gpu
def test_afn_model(deeprec_resource_path):
    # Note: must split the test from models like xDeepFM, which use tf.v1 to construct the Graph
    # tf.config.run_functions_eagerly(True)
    data_path = os.path.join(deeprec_resource_path, "xdeepfm")
    yaml_file = os.path.join(data_path, "AFN.yaml")
    if not os.path.exists(yaml_file):
        download_deeprec_resources(
            "https://recodatasets.z20.web.core.windows.net/deeprec/",
            data_path,
            "xdeepfmresources.zip",
        )
    hparams = prepare_hparams(yaml_file,FEATURE_COUNT=1000, FIELD_COUNT=10)

    model = AFN(field_cnt=hparams.FIELD_COUNT, feature_cnt=hparams.FEATURE_COUNT,
                embedding_size=hparams.embedding_size, ltl_hidden_size = hparams.ltl_hidden_size,
                afn_dnn_hidden_units = hparams.afn_dnn_hidden_units,
                ensamble_dnn_units = hparams.ensamble_dnn_units,
                l2_reg = hparams.l2_reg,
                dnn_dropout = hparams.dnn_dropout,
                dnn_activation = hparams.dnn_activation,
                output_activation = hparams.output_activation,
                use_bn = hparams.use_bn
    )


    model.train_model(
            logdir=osp.join(data_path, "tf_sum"),
              train_data=osp.join(data_path, "synthetic_part_0"),
              validation_data=osp.join(data_path, "synthetic_part_1"),
              start_learning_rate=float(hparams.start_learning_rate),
              end_learning_rate=float(hparams.end_learning_rate),
              decay_power=float(hparams.decay_power),
              batch_size=int(hparams.batch_size),
              epochs=2,
              steps_per_epoch=40,
              validation_steps=20)