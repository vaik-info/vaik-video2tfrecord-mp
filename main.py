import argparse
import os
import glob
import random
import shutil
import tqdm
from multiprocessing import Process, cpu_count

import tensorflow as tf
import io_tfrecords

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def split(a_list, split_num):
    result_list = []
    for split_index in range(split_num):
        result_list.append([])
    for index, an_elem in enumerate(a_list):
        result_list[index % split_num].append(an_elem)
    return result_list


def write(video_path_label_list, output_dir_path, classes, proc_num, records_prefix_index):
    tf_record_writer = tf.io.TFRecordWriter(
        os.path.join(output_dir_path, f"dataset.tfrecords-{records_prefix_index}{proc_num:03d}"))
    for video_path, class_label in tqdm.tqdm(video_path_label_list):
        example = io_tfrecords.video2tfrecords(video_path, classes.index(class_label))
        tf_record_writer.write(example.SerializeToString())
    tf_record_writer.close()


def main(input_dir_path, input_classes_path, output_dir_path, records_prefix_index, cpu_count):
    os.makedirs(output_dir_path, exist_ok=True)

    classes = []
    with open(input_classes_path) as f:
        for line in f:
            classes.append(line.strip())

    video_path_list = []
    for class_label in classes:
        for video_path in glob.glob(os.path.join(input_dir_path, class_label, '*.mp4')):
            video_path_list.append((video_path, class_label))
    random.shuffle(video_path_list)

    split_video_path_label_list = split(video_path_list, cpu_count)

    processes = []
    for proc_num in range(len(split_video_path_label_list)):
        p = Process(target=write, args=(
        split_video_path_label_list[proc_num], output_dir_path, classes, proc_num, records_prefix_index))
        processes.append(p)

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    shutil.copy2(input_classes_path, os.path.join(output_dir_path, os.path.basename(input_classes_path)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main')
    parser.add_argument('--input_dir_path', type=str, default='~/.vaik-utc101-video-classification-dataset/train')
    parser.add_argument('--input_classes_path', type=str,
                        default='~/.vaik-utc101-video-classification-dataset/ucf101_labels.txt')
    parser.add_argument('--output_dir_path', type=str,
                        default='~/.vaik-utc101-video-classification-dataset_tfrecords/train')
    parser.add_argument('--records_prefix_index', type=str, default='00')
    parser.add_argument('--cpu_count', type=int, default=cpu_count())
    args = parser.parse_args()

    args.input_dir_path = os.path.expanduser(args.input_dir_path)
    args.input_classes_path = os.path.expanduser(args.input_classes_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    main(**args.__dict__)
