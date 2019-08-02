from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import src.util.create_dataset as cd


def main():
    lsp = True
    lsp_e = False
    mpii = False
    dir_name = '/home/valentin/Code/ADL/human-pose-estimation/datasets/'
    if lsp:
        tfrecords_file_name = dir_name + 'lsp_train.tfrecords'
        file_name_pairs = cd.get_filename_pairs_lsp()
        cd.create(tfrecords_file_name, file_name_pairs[:8], dataset="lsp")
        print("creating dataset with", len(file_name_pairs),"entries")
    if lsp_e:
        tfrecords_file_name = dir_name + 'lsp_ext.tfrecords'
        file_name_pairs = cd.get_filename_pairs_lspe()
        cd.create(tfrecords_file_name, file_name_pairs, dataset="lsp_ext")
        print("creating dataset with", len(file_name_pairs), "entries")
    if mpii:
        tfrecords_file_name = dir_name + 'mpii.tfrecords'
        file_name_pairs = cd.get_filename_pairs_mpii()
        cd.create(tfrecords_file_name, file_name_pairs, dataset="mpii")
        print("creating dataset with", len(file_name_pairs),"entries")


if __name__ == '__main__':
    main()
