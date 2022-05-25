# traveloka-ocr-ml
Machine Learning engineer team repository of Traveloka OCR Project.

## Create lmdb dataset before training.
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