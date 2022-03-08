#!/usr/bin/env bash
mkdir data
wget https://endocv2021.s3.eu-west-2.amazonaws.com/trainData_EndoCV2021_21_Feb2021-V2.zip
unzip trainData_EndoCV2021_21_Feb2021-V2.zip -d data
rm trainData_EndoCV2021_21_Feb2021-V2.zip

# fix mistaken name
mv 'data/trainData_EndoCV2021_21_Feb2021-V2/data_C3/images_C3/C3_EndoCV2021_00489].jpg' 'data/trainData_EndoCV2021_21_Feb2021-V2/data_C3/images_C3/C3_EndoCV2021_00489.jpg'
python prepare_endo_data.py
