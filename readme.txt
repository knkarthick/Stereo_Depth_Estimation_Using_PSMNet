This repository contains the code in PyTorch for "Pyramid Stereo Matching Network(https://arxiv.org/abs/1803.08669)" paper 
(CVPR 2018) by [Jia-Ren Chang](https://jiarenchang.github.io/) and [Yong-Sheng Chen](https://people.cs.nctu.edu.tw/~yschen/).

Various changes pre-processing steps were added to the code by Sloka R Anugu. 
Additional Experiments with Transfer Learning were undertaken to replace the original feature extractor of the PSMnet. 
The best experiment, ResNet-50_1, is included in the code.

Getting Started:

1. Set up the env with by running the below command:
conda env create --name envname --file=pyramid_test.yml
conda activate envname

2. Download the datasets:
Download RGB cleanpass images and its disparity for three subset: FlyingThings3D, Driving, and Monkaa.
Put them in the 'dataset' folder.
And rename the folder as: "driving_frames_cleanpass", "driving_disparity", "monkaa_frames_cleanpass", "monkaa_disparity", "frames_cleanpass", "frames_disparity".
Datasets can be downloaded from: https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html


Please note: For this experiment, only the Driving Dataset was used.


3. Run main.py 

Sample command: 
python main.py --maxdisp 192 \
               --model stackhourglass \
               --datapath (your scene flow data folder)\
               --epochs 10 \
               --loadmodel (optional)\
               --savemodel (path for saving model)

Please note: To use ResNet-50 as the feature extractor, please go to the models folder and open submodule.py script.
             Comment the first feature_extraction class and uncomment the second feature_extraction class

4. Download KITTI 2012 and KITTI 2015 datasets from the below link and save them in two separate directories:
KITTI 2012 - http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo
KITI 2015 - http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo


5. Run finetune.py on KITTI 2012 and KITTI 2015 one after the other

Sample Command
python finetune.py --maxdisp 192 \
                   --model stackhourglass \
                   --datatype 2015 \
                   --datapath (KITTI 2015 training data folder) \
                   --epochs 200 \
                   --loadmodel (pretrained PSMNet trained on SceneFlow  data) \
                   --savemodel (path for saving model)

Please note: 
- As ground truth for KITTI test data is not available, Train data of KITTI was used as training, validation 
and test data. 
- To use ResNet-50 as the feature extractor, please go to the models folder and open submodule.py script.
Comment the first feature_extraction class and uncomment the second feature_extraction class

6. To run trained models on the KITTI TEST data for outputs, run python submission.py

Sample Command:
python submission.py --maxdisp 192 \
                     --model stackhourglass \
                     --KITTI 2015 \
                     --datapath (KITTI 2015 test data folder) \
                     --loadmodel (Trained finetuned PSMNet Model) \

7. Run test_img.py to test a stereo pair

Sample Command:
python Test_img.py --loadmodel (finetuned PSMNet) --leftimg ./left.png --rightimg ./right.png
