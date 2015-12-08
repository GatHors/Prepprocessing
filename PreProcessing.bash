#!/bin/bash
cd $(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
filepath=$1
filename=$(basename "$filepath")
videoname="${filename%.*}"
source /etc/profile
mkdir /home/gathors/proj/v-opencv/preimport/$videoname
ffmpeg -i $filepath -r 1 -f image2 /home/gathors/proj/v-opencv/preimport/$videoname/%06d.jpg
/home/gathors/proj/v-opencv/hibImport.sh -f /home/gathors/proj/v-opencv/preimport/$videoname admin/input/input.hib
#usage: hibImport.jar [options] <image directory> <output HIB>
# -f,--force        force overwrite if output HIB already exists
gsutil cp -R /home/gathors/proj/v-opencv/preimport/$videoname gs://gathors
sudo rm -r /home/gathors/proj/v-opencv/preimport/$videoname
#run hadoop feature extraction

