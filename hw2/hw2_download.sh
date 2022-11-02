#!/bin/bash

wget -P "saved_models/" "https://www.dropbox.com/s/bhb2oxb2daqjgs1/DANN_usps.zip"
unzip -q "saved_models/DANN_usps.zip" -d "saved_models/"

wget -P "saved_models/" "https://www.dropbox.com/s/fscsj99a74x07nd/DANN_svhn.zip"
unzip -q "saved_models/DANN_svhn.zip" -d "saved_models/"

wget -P "saved_models/" "https://www.dropbox.com/s/e1y2ipf48m4p7sz/DCGAN.zip"
unzip -q "saved_models/DCGAN.zip" -d "saved_models/"

wget -P "saved_models/" "https://www.dropbox.com/s/v1y0d19py542uf5/DDPM_EMA.zip"
unzip -q "saved_models/DDPM_EMA.zip" -d "saved_models/"