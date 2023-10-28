#!/bin/bash

if [[ ! -f ~/.kaggle/kaggle.json ]]; then
  echo -n "Kaggle username: "
  read USERNAME
  echo
  echo -n "Kaggle API key: "
  read APIKEY

  mkdir -p ~/.kaggle
  echo "{\"username\":\"$USERNAME\",\"key\":\"$APIKEY\"}" > ~/.kaggle/kaggle.json
  chmod 600 ~/.kaggle/kaggle.json
fi

pip install kaggle --upgrade
kaggle datasets download -d dansbecker/cityscapes-image-pairs
unzip cityscapes-image-pairs.zip
rm cityscapes-image-pairs.zip
mv cityscapes_data/train training_data/
mv cityscapes_data/val training_data/
rm -rf cityscapes_data
