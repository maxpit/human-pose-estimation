# Hybrid Method for 3D Human Pose Estimation

This is the code written for the [Advanced Deep Learning for Computer Vision](https://dvl.in.tum.de/teaching/adl4cv-ss19/) course project offered by TU Munich. 

A detailed report about our project can be found [here](report/report.pdf).

The project is a hybrid model using approaches from different papers. Our main pipeline is based on the code from Kanazawa et. al. that can be found [here](https://github.com/akanazawa/hmr).
We reused a lot of code from their repository but changed everything to use tensorflow 2.0 and python 3.5, furthermore we implemented additional loss functions and modified the network architecture. The following list shows which files we modified and what we changed in each of them.

### Changerecord
#### New files
 - `src/predictor.py`
	Used for inference, was mainly copied from our reworked src/trainer.py
 - `src/util/create_dataset.py`
	Used to create the tfrecords files for the different datasets.
 - `preview.py`
	A webcam demo that uses a pretrained model and visulizes the results
 - `train.py`
	Used for training the network
 - `validate_checkpoint.py`
	Can be used get validation scores on the validation set for a given checkpoint
 - All files in `src/visualizations` which can be used to visualize the different datasets

#### Files changed a lot
 - `src/trainer.py`
	Due to changes in the dataloader and the change to eager execution and tf2 most of the code here needed to be rewritten
 - `src/models.py`
	Added the critic network
 - `src/ops.py`
	Added the mesh reprojection loss
 - `src/tf_smpl/projection.py`
	Added mesh reprojection

#### Files changed a little
 - `src/config.py`
	Added new config parameters and removed unused ones
 - `src/data_loader.py`
	Changed to the tensorflow Dataloader API and made changes to accomodate for the additional input data needed (segmentation gt)
 - `src/util/data_utils.py`
	Added utils for added segmentation gt

#### No significant changes except for updating to tf2
 - All files in `src/tf_smpl/` except projection.py
 - `src/util/renderer.py`
 - `src/util/image.py`

## Requirements
- Python 3.5
- [TensorFlow](https://www.tensorflow.org/) tested with version 2.0

## Installation 
### Create the environment
use `conda env create -f environment.yml` to install a new conda environment from the environment.yml file

use `conda activate hpe` to activate the new environment.

use the fork on https://github.com/vstarlinger/opendr and follow the installation procecdure in the readme in order to install opendr and chumpy

first go to the chumpy folder and install it using `python setup.py install`
then go to the opendr folder and install it using `python setup.py install`

use the fork on https://github.com/vstarlinger/SMPL to preprocess the SMPL models from the [End to end recovery of human shape and pose](https://akanazawa.github.io/hmr/) paper and save it in the same folder as the original models with filename 'model' (or change the config to point to the correct model).

### Install TensorFlow
With GPU:
```
pip install tensorflow-gpu==2.0.0-beta1
```
Without GPU:
```
pip install tensorflow==2.0.0-beta1
```

## Demo
The demo code for this project uses the computers webcam and predicts the 3D human pose from the input image using a pre-trained model. It automatically runs in fullscreen mode.
If you need the pretrained model (~700 MB) please contact one of the authors.

It can be controlled using the following keyboard commands:
 - s: display the skeleton on top of the input image
 - m: display the mesh on top of the input image (default)
 - b: display the mesh on top of the input image as well as the mesh rotated by 60 degrees
 - r: display only the rotated version of the mesh
 - ESC: end the program

## Training

For training first the datasets need to be created, this can be done by first downloading the LSP and LSP extended datasets and then using the create\_datasets.py file to generate the tfrecords files in the desired dataset directories specified in the config file.

To get the 3D data for training the critic network please follow the instructions provided [here](https://github.com/akanazawa/hmr/blob/master/doc/train.md#mosh-data).

After configuring the tfrecord files, the training can be started by using the `train.py` file. The parameters for the training can either be configured by editing the `src/config.py` file or by using the corresponding flags.

```
python -m train --num_epochs=120
```

## Authors
Maximilian Pittner
max.pittner _ ät _ tum.de
Valentin Starlinger
valentin.starlinger _ ät _ tum.de

