# vaik-video2tfrecord-mp
Parallel convert from mp4 format to tfrecord format for video classification.

## Example

![vaik-video2tfrecord-mp](https://github.com/vaik-info/vaik-video2tfrecord-mp/assets/116471878/de292c5a-43fd-4584-b42e-0c08a8a13322)

## Usage

```shell
pip install -r requirements.txt
python main.py --input_dir_path ~/.vaik-utc101-video-classification-dataset/train \
                --input_classes_path ~/.vaik-utc101-video-classification-dataset/ucf101_labels.txt \
                --output_dir_path ~/.vaik-utc101-video-classification-dataset_tfrecords/train \
                --records_prefix_index 00 \
                --cpu_count 32
```

## Output

![Screenshot from 2023-07-04 14-24-00](https://github.com/vaik-info/vaik-video2tfrecord-mp/assets/116471878/71ece23e-e697-4db3-b0ac-a1b56af230de)

## Parse Example

```python
import os
import glob
import tensorflow as tf


def parse_tfrecord_fn(example):
    feature_description = {
        'video': tf.io.FixedLenFeature([], tf.string),
        'shape': tf.io.FixedLenFeature([4], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, feature_description)
    video = tf.io.parse_tensor(example['video'], out_type=tf.uint8)
    shape = example['shape']
    video = tf.reshape(video, shape)
    label = example['label']
    return video, label


filenames = glob.glob(os.path.expanduser("~/.vaik-utc101-video-classification-dataset_tfrecords/train/*"))
raw_dataset = tf.data.TFRecordDataset(filenames)
parsed_dataset = raw_dataset.map(parse_tfrecord_fn)

for video, label in parsed_dataset.take(2):
    print("Video shape:", video.shape)
    print("Label:", label.numpy())
```