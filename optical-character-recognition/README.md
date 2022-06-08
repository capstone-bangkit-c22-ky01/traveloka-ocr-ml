# traveloka-ocr-ml
Optical Character Recognition

## Training Process
### Dataset
We retrieve data that has been cutted by Object Detection model and then manually annotating the images into txt file. We took 5 information from the KTP in the form of NIK, Name, Gender, Marital Status, and Nationality which would be used as a class. 

### Data Preprocessing
After annotating, we preprocessed the images using **CV2** like below:
- Noise removal to remove noise i.e. unwanted dots and strips in ID card
- Grayscaling reduce image dimensionality
- Thresholding Onsu to project pixel value into binary (0 or 255)
  
Noise Removal for removing the noise, Thresholding Otsu, Grayscaling
| Demo images | Preprocessed | 
| ---         |     ---      | 
| <img src="./misc/data/nik.png" width="300" height="30">    |  <img src="./misc/data_processed/nik.png" width="300" height="30">   |  
| <img src="./misc/data/name.png" width="300" height="30">    |  <img src="./misc/data_processed/name.png" width="300" height="30">   |  
| <img src="./misc/data/sex.png" width="300" height="30">    |  <img src="./misc/data_processed/sex.png" width="300" height="30">   |  
| <img src="./misc/data/marital_status.png" width="300" height="30">    |  <img src="./misc/data_processed/marital_status.png" width="300" height="30">   |  
| <img src="./misc/data/nationality.png" width="300" height="30">    |  <img src="./misc/data_processed/nationality.png" width="300" height="30">   |  

### Create LMDB Dataset before training.
After preprocessing the images and annotating, we need to convert the datasets into **LMDB format** like below:
```
pip3 install fire
python3 create_lmdb_dataset.py --inputPath data/ --gtFile data/gt.txt --outputPath result/
```
The structure of data folder as below.
```
data
├── gt.txt
└── train_data
    ├── image_1.png
    ├── image_2.png
    ├── image_3.png
    └── ...
```
At this time, `gt.txt` should be `{imagepath}\t{label}\n` <br>
For example
```
train_data/image_1.png 195153737373737
train_data/image_2.png GOOGLE BANGKIT ACADEMY
train_data/image_3.png qwertyuiop
...
```

### Modelling
We deeply adapt the model architecture from [ClovaAI](https://github.com/clovaai/deep-text-recognition-benchmark), which we create the architecture like below:
- Feature Extractor using VGG16 or ResNet
- Loss Function using CTC Loss
<p align="center">
    <img src="./misc/vgg16.png" alt="VGG16 Architecture" width="550" style="vertical-align:middle">
</p>
Implementation of VGG16 (image from [Researchgate.net] (https://www.researchgate.net/figure/Fig-A1-The-standard-VGG-16-network-architecture-as-proposed-in-32-Note-that-only_fig3_322512435))

## Demo
| Demo images  | prediction result |
| ---            |          --- |
| <img src="./misc/data/nik.png" width="300" height="30">       |  3329091212121059   |
| <img src="./misc/data/name.png" width="300" height="30">     |  BORUTO   |
| <img src="./misc/data/sex.png" width="300" height="30">     |  LAKI-LAKI   |
| <img src="./misc/data/marital_status.png" width="300" height="30">     |  BELUM KAWIN   |
| <img src="./misc/data/nationality.png" width="300" height="30">     |  WNI   | 

## References
