#!/bin/bash

sudo apt update
sudo apt install -y poppler-utils

pip -r install requirements.txt

echo "######################################"
echo "OK"
echo "######################################"
