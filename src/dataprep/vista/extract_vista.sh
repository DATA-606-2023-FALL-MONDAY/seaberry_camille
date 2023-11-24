#!/usr/bin/env bash

# very large zip file downloaded from Mapillary site
# extract json files, then use those to get list of which ones have cctv
# then extract just those images
zipfile="./data/vista/mapvistas.zip"
unzip -j -n $zipfile "training/v2.0/polygons/*.json" -d ./data/vista/json

# using grep, get an array of json files that match the string "cctv"
cctvjson=($(grep -l "cctv" ./data/vista/json/*.json))
# echo length of cctvjson
echo ${#cctvjson[@]}

# loop over files in cctvjson array and extract the image id
for jsonfile in "${cctvjson[@]}"
do
  id=$(basename "$jsonfile" .json)
  img="training/images/$id.jpg"
  mask="training/v2.0/instances/$id.png"
  unzip -j -n -q $zipfile $img -d ./data/vista/images
  # don't need mask images since json annotations work on roboflow
  # unzip -j -n -q $zipfile $mask -d ./data/vista/masks
done

ls -1 ./data/vista/images | wc -l

# after extracting images that have cctv, manually upload to roboflow as semantic project
# then delete other classes & duplicate as object detection
# annotation json file downloaded from mapillary2coco repo--would rather do it programmatically
# but i'm too tired at this point
# https://github.com/Luodian/Mapillary2COCO