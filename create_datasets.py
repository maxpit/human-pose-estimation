from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from os.path import join, dirname
from absl import flags
import src.util.create_dataset as cd
import src.data_config

def main(config):
    if config.create_lsp:
        tfrecords_file_name = join(dir_name, 'lsp_train.tfrecords')
        file_name_pairs = cd.get_filename_pairs_lsp(config)
        cd.create(config,tfrecords_file_name, file_name_pairs[:1000], dataset="lsp")
        print("creating dataset with", len(file_name_pairs),"entries")
    if config.create_lsp_val:
        tfrecords_file_name = join(dir_name, 'lsp_val.tfrecords')
        file_name_pairs = cd.get_filename_pairs_lsp(config)
        cd.create(config,tfrecords_file_name, file_name_pairs[1000:], dataset="lsp")
        print("creating dataset with", len(file_name_pairs),"entries")
    if config.create_lsp_ext:
        tfrecords_file_name = join(dir_name, 'lsp_ext.tfrecords')
        file_name_pairs = cd.get_filename_pairs_lspe(config)
        cd.create(config,tfrecords_file_name, file_name_pairs, dataset="lsp_ext")
        print("creating dataset with", len(file_name_pairs), "entries")
    if config.create_mpii:
        tfrecords_file_name = join(dir_name, 'mpii.tfrecords')
        file_name_pairs = cd.get_filename_pairs_mpii(config)
        cd.create(config,tfrecords_file_name, file_name_pairs, dataset="mpii")
        print("creating dataset with", len(file_name_pairs),"entries")


if __name__ == '__main__':
    data_config = flags.FLAGS
    data_config(sys.argv)
    main(data_config)
