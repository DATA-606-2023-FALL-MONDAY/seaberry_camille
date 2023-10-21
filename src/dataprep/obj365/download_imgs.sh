#!/bin/sh

# train has 50 patches, val has 43: 0-15 in v1, 16-43 in v2

base="https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86"
split=$1
outdir=${2:-"data/obj365"}

# nsamp=$2
oldjson="zhiyuan_objv2_$split.json"
newjson="data/obj365/$split-ann.json"

echo "split: $split"

# download json of metadata--annotations, labels, paths to images
# img_url="$base/$split/patch$i.tar.gz"
if [[ $split = "train" ]]; then
  url="$base/$split/zhiyuan_objv2_$split.tar.gz"
  fn="$split-ann.tar.gz"
  if [ ! -f $fn ]; then
    curl $url --output $fn
    tar xvf $fn
  fi
  if [ ! -f $newjson ]; then
    mv $oldjson $newjson
  fi
  # download patches
elif [[ $split = "val" ]]; then
  url="$base/$split/$oldjson"
  if [ ! -f $newjson ]; then
    curl $url --output $newjson
  fi
else
  echo "wrong split type"
  exit 1
fi 

# download tar files of images
download_tars(){
  v=$1
  i=$2
  patch="patch$i.tar.gz"
  dl="$base/$split/images/$v/$patch"
  tarout="$outdir/images/$v/$patch"
  if [ ! -f $tarout ]; then
    curl -s -w "%{filename_effective}\n" $dl --output $tarout
  fi
}

if [[ $split = "val" ]]; then
  mkdir -p obj365/images/{v1,v2}
  for i in {0..15}; do
    download_tars v1 $i  
  done
  for i in {16..43}; do
    download_tars v2 $i  
  done
fi
wait


# fix annotations into proper coco format
# python sahi_cleanup.py $split
# extract images with relevant categories
# python prep_obj.py $split 


