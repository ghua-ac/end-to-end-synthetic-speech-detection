# End-to-End Synthetic Speech Detection

# Important Notice (Oct. 2021)
**The results reported in our [paper](https://ieeexplore.ieee.org/document/9456037) were based on Windows system, while we recently found that the execution of the same repo and dataset on Linux yielded different results, using the pretrained [models](https://github.com/ghuawhu/end-to-end-synthetic-speech-detection/tree/main/pretrained):**

- **Res-TSSDNet ASVspoof2019 eval EER: 1.6590%;**
- **Inc-TSSDNet ASVspoof2019 eval EER: 4.0384%.**

**We have identified issues of the package *soundfile* on Windows when writing and reading flac files, but this problem does not exist on Linux for the same package. The similar problem has been pointed out [here](https://github.com/ghuawhu/end-to-end-synthetic-speech-detection/issues/2).**

## About
We present two light-weight neural network models, termed time-domain synthetic speech detection net (TSSDNet), having the classic ResNet and Inception Net style structures (Res-TSSDNet and Inc-TSSDNet), for end-to-end synthetic speech detection. They achieve the state-of-the-art performance in terms of equal error rate (EER) on ASVspoof 2019 challenge and are also shown to have promising generalization capability when tested on ASVspoof 2015. 

## Dataset
- ASVspoof 2019 LA partition. [link](https://datashare.ed.ac.uk/handle/10283/3336)
- ASVspoof 2015. [link](https://datashare.ed.ac.uk/handle/10283/853)
  
1. ASVspoof 2019 train set is used for training;
2. ASVspoof 2019 dev set is used for model selection;
3. ASVspoof 2019 eval set is used for testing;
4. ASVspoof 2015 eval set is used for cross-dataset testing.

## Model Architecture
<center><img src="https://github.com/ghuawhu/end-to-end-synthetic-speech-detection/raw/main/imgs/1.png" width="500"></center>

## Main Results
The two models with 1.64% and 4.04% eval EER (below), and their train logs, are provided in folder pretrained.

<center><img src="https://github.com/ghuawhu/end-to-end-synthetic-speech-detection/raw/main/imgs/2.png" width="500"></center>

Fixing all hyperparameters, the distribution of the lowest dev (and the corresponding eval) EERs among 100 epochs, trained from scratch (below):

<center><img src="https://github.com/ghuawhu/end-to-end-synthetic-speech-detection/raw/main/imgs/3.png" width="500"></center>

## Usage
### Data Preparation 
```
ASVspoof15&19_LA_Data_Preparation.py
```
It generates 
1) equal-duration time domain raw waveform
2) 2D log power of constant Q transform

from ASVspoof2019 and ASVspoof2015 official datasets, respectively. The calculation of CQT is adopted from [Li et al. ICASSP 2021](https://github.com/lixucuhk/ASV-anti-spoofing-with-Res2Net).

### Training 
```
train.py
```
It supports training using 
1) standard cross-entropy vs weighted cross-entropy
2) standard train loader vs mixup regularization
3) 1D raw waveforms vs 2D CQT feature
4) ASVspoof 2019 training set vs ASVspoof 2015 training set

A train log will be generated, and trained models per epoch will be saved.

### Testing
```
test.py
```
It generates softmax accuracy, ROC curve, and EER.

## Citation Information
G. Hua, A. B. J. Teoh, and H. Zhang, “Towards end-to-end synthetic speech detection,” IEEE Signal Processing Letters, vol. 28, pp. 1265–1269, 2021. [arXiv](https://arxiv.org/abs/2106.06341) | [IEEE Xplore](https://ieeexplore.ieee.org/document/9456037)
