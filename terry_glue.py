# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os


import Terry_toolkit as tkit

from transformers import DataProcessor, InputExample, InputFeatures
# from transformers import glue_processors as processors
# from ...file_utils import is_tf_available

# if is_tf_available():
#     import tensorflow as tf

logger = logging.getLogger(__name__)


class TerryProcessor(DataProcessor):
    """Processor for 自定义数据集 10分类"""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(tensor_dict['idx'].numpy(),
                            tensor_dict['sentence'].numpy().decode('utf-8'),
                            None,
                            str(tensor_dict['label'].numpy()))

    def get_train_examples(self, data_dir):
        """See base class."""
        file_path = os.path.join(data_dir,"train.json")
        tjosn=tkit.Json(file_path=file_path).load()
        return self._create_examples(tjosn, 'train')
    def get_dev_examples(self, data_dir):
        """See base class."""
        file_path = os.path.join(data_dir,"dev.json")
        tjosn=tkit.Json(file_path=file_path).load()
        return self._create_examples(tjosn, 'dev')

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3","4","5","6","7","8","9"]
 

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line['sentence']
            label = line['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

glue_tasks_num_labels = {
    "terry": 10,

}

glue_processors = {
    "terry": TerryProcessor,

}

glue_output_modes = {
    "terry": "classification",
}
