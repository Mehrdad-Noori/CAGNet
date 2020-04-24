# CAGNet
This repository contains the Tensorflow implementation of our paper "CAGNet: Content-Aware Guidance for Salient Object Detection". The paper can be found at: [[Sciencedirect]](https://www.sciencedirect.com/science/article/abs/pii/S0031320320301072) [[arXiv]](https://arxiv.org/abs/1911.13168)


### Requirements
- [numpy 1.17.4](https://numpy.org/)
- [tensorflow 2.1.0](https://www.tensorflow.org/)
- [scipy 1.4.1](https://www.scipy.org/)
- [pillow 7.0.0](https://pillow.readthedocs.io/)
- [opencv-python](https://github.com/skvark/opencv-python)

### Usage
1- Clone the repository
```
git clone https://github.com/Mehrdad-Noori/CAGNet
cd CAGNet
```
2- If you want to train the model, download the following dataset and unzip it into `data` folder.
- [DUTS Dataset: Training](http://saliencydetection.net/duts/)

3- To run the training, set the arguments or use the default settings: 

```
python train.py --backbone_model 'ResNet50' --batch_size 10 --save_dir 'save'
```
The backbone_model can be one of the following options: `VGG16`, `ResNet50`, `NASNetMobile` or `NASNetLarge` 

4- To generate saliency maps:
```
python predict.py --model '/path/to/trained/model' --input_dir /path/to/input/images/directory --save_dir 'save'
```
You can also download and use [our pre-trained models](#pre-trained-models--pre-computed-saliency-maps)


5- Evaluation code

You can use [this toolbox](https://github.com/Mehrdad-Noori/Saliency-Evaluation-Toolbox) to compute different saliency measures such as E-measure, Weighted F-measure, MAE, PR curve ...

### 


### Pre-trained models & pre-computed saliency maps
We provide the pre-trained model and pre-computed saliency maps for DUTS-TE, ECSSD, DUT-OMRON, PASCAL-S, and HKU-IS datasets.

- CAGNet-V (VGG16 backbone): [[pre-trained model]](https://drive.google.com/drive/folders/1V8L5x5FjDrBU04uueVnHYJi7W8E8KGdN?usp=sharing) - [[saliency maps]](https://drive.google.com/open?id=1T2qB-axQOSXPT2XOQ_zfBFDdIlIbgpsf)
- CAGNet-R (ResNet50 backbone): [[pre-trained model]](https://drive.google.com/drive/folders/1a763tL98Z3DUmpl3BisoRh5FWafaV4i1?usp=sharing) - [[saliency maps]](https://drive.google.com/open?id=1YIJTPShV93PvNvz-LZP4NMVbwbFNHBBM)
- CAGNet-M (NASNet Mobile backbone): [[pre-trained model]](https://drive.google.com/drive/folders/13inkoc0kj5lbX0EphWgfSweRX1uqWQ3A?usp=sharing) - [[saliency maps]](https://drive.google.com/open?id=1T3W-lvQpqJrD8JzfQm4K2_PQ4caOBWir)
- CAGNet-L (NASNet Large backbone): [[pre-trained model]](https://drive.google.com/drive/folders/12mo-8qYsDSLkzPAGHydbgW0Ibmr2rm1K?usp=sharing) - [[saliency maps]](https://drive.google.com/open?id=1C-lP99h4W_0Gx1QKQ9HkiX8xhP38Hb7p)

### Quantitative Comparison

![image](https://github.com/Mehrdad-Noori/CAGNet/blob/master/figures/quantitative.jpg)


### Qualitative Comparison

![image](https://github.com/Mehrdad-Noori/CAGNet/blob/master/figures/qualitative.jpg)

### Any problems?

Please feel free to contact me, or raise an issue if you encounter any problems.

### Citation
```
@article{mohammadi2020cagnet,
  title={CAGNet: Content-Aware Guidance for Salient Object Detection},
  author={Mohammadi, Sina and Noori, Mehrdad and Bahri, Ali and Majelan, Sina Ghofrani and Havaei, Mohammad},
  journal={Pattern Recognition},
  pages={107303},
  year={2020},
  publisher={Elsevier}
}
```
