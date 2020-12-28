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

- CAGNet-V (VGG16 backbone) - model: [GoogleDrive](https://drive.google.com/drive/folders/1r0Ayj3zqCAoIw-2Got5SjQHQTHUcuo2S?usp=sharing), [baidu](https://pan.baidu.com/s/1HoqxPXseoBOYQCmuiDuAHg) (extraction code: bne8)-  saliency maps: [GoogleDrive](https://drive.google.com/file/d/1Xi6KbWu66cTQXR3pdqNea8j-p2f60bkA/view?usp=sharing), [baidu](https://pan.baidu.com/s/1ECLysn6MQIFkMpbnVUI_EA) (extraction code: m3id)
- CAGNet-R (ResNet50 backbone) - model: [GoogleDrive](https://drive.google.com/drive/folders/1KguJGzwKDu8VcsqMGwXPDwo3vxr7BmSE?usp=sharing), [baidu](https://pan.baidu.com/s/1OwLa42wMQ86pcnSC0wBALQ) (extraction code: bp22) - saliency maps: [GoogleDrive](https://drive.google.com/file/d/15uBSkjTvbXWG1iy7toHa5QPC5BY1zaJO/view?usp=sharing), [baidu](https://pan.baidu.com/s/1nHWKCsQrjYrP86aKbHIyJQ) (extraction code: kv9w)
- CAGNet-M (NASNet Mobile backbone) - model: [GoogleDrive](https://drive.google.com/drive/folders/1AiPBYTE9D2Y596tyWidfrup255Fzly7P?usp=sharing), [baidu](https://pan.baidu.com/s/1l3OncGKXkFR9Fq8hg2iccw) (extraction code: 8kx6) - saliency maps: [GoogleDrive](https://drive.google.com/file/d/10ZKask4uu_tL3LExpBTg9doTwTs4d0rQ/view?usp=sharing), [baidu](https://pan.baidu.com/s/145m4SUYaJBOvgfVJNO9yBg) (extraction code: q8v1)
- CAGNet-L (NASNet Large backbone) - model: [GoogleDrive](https://drive.google.com/drive/folders/1s6nKDeufwkH0pvxiMAw7lhSMBJHFZuG_?usp=sharing), [baidu](https://pan.baidu.com/s/1ktFoP9EDI5eGBkL4lih7kA) (extraction code: idh8) - saliency maps: [GoogleDrive](https://drive.google.com/file/d/14fqFZORttFidBBHytJggrcdCdSyVc0jv/view?usp=sharing), [baidu](https://pan.baidu.com/s/1TtOP3n4CnheCo9WjIgu_nA) (extraction code: kety)

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
