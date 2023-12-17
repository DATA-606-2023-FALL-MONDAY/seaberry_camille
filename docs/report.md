# Report: Re-surveilling surveillance
Camille Seaberry

Prepared for UMBC Data Science Master’s Degree Capstone with Dr. Chaojie
Wang, Fall 2023

- Code: https://github.com/DATA-606-2023-FALL-MONDAY/seaberry_camille
- Presentation:
  - Interactive: https://camille-s.github.io/capstone_pres
  - Static:
    https://camille-s.github.io/capstone_pres/seaberry_slides.pdf
- App:
  - Deployment: https://camilleseab-surveillance.hf.space/
  - Code:
    https://huggingface.co/spaces/camilleseab/surveillance/tree/main
- Github: https://github.com/camille-s

## Background

Police surveillance cameras in Baltimore—many of which are situated at
street intersections that make up a spatial network by which people move
through the city—form one of many layers of state surveillance imposed
upon residents. These cameras are often clearly visible at a distance,
unlike less obvious layers operated by wannabe-state actor-vendors like
Amazon (private Ring cameras subsidized by Amazon and distributed by
police departments such as Baltimore) or Axon (with a monopoly over
body-worn cameras, a band aid offered to counter police violence, and
Tasers, a “less-lethal” potentially lethal high-tech weapon). This
visibility, sometimes including blaringly bright blue lights, creates an
announcement of the fact of being watched. Yet there is little
documentation and even less direct control or oversight of this
landscape, and even crowdsourced data sources like OpenStreetMap have
very little of this landscape recorded. These histories are laid out in
research such as Browne (2015).

I would like to build upon my final project in Data 690: Deep Learning
(Spring 2023). I attempted to recreate aspects of two papers (Turtiainen
et al. (2021); and Sheng, Yao, and Goel (2021)). Both of these papers
train deep learning models on several thousand photos of urban
streetscapes, including those batch downloaded from Google Street View.
Sheng, Yao, and Goel (2021) make their download script available, and
use Baltimore as one of their test cities, so I was able to use a sample
of their images directly. In addition, I used the Objects365 dataset
(Shao et al. (2019)), the only one of the standard image datasets I
could find that specifically had surveillance cameras annotated. Using
these images and a few predefined models from Facebook Research’s
Detectron2 library (Wu et al. (2019)), I trained several neural networks
to identify the locations of surveillance cameras in these images. With
some success, I then developed models with PyTorch to categorize cameras
as directed or global, an additional annotation in the Sheng, Yao, and
Goel (2021) dataset.

The major purposes of those two papers involved mapping the locations of
cameras after detecting them. Because the Street View images can be
downloaded based on their coordinates, once a camera is detected in an
image, its location is known.

For this project, my original goal was to improve upon the models I
used, including introduction of more predefined models (adding YOLO,
among others), finer tuning of classification of camera type (including
possibly adding automated license plate readers), more concerted
sampling of intersections in Baltimore, and updated images. Over the
course of the semester, I developed several variations on two model
types for detection and one for classification to work with decent
accuracy.

Longer term, I’d like to use these models to map locations of cameras to
study the landscape of surveillance. I would like to do some amount of
spatial analysis overlaying camera locations and socio-economic /
demographic data to understand any patterns in this landscape and the
potential burdens of surveillance on marginalized communities in
Baltimore.

Therefore my **major research questions are:**

1.  How accurately can deep learning models detect surveillance cameras
    in street images?
2.  How accurately can deep learning models classify types of
    surveillance cameras?

## Data

This project uses several non-tabular data sources. As the main part of
this project is object detection in images, the images’ pixel data will
become the features. In the first task (detection), the target is the
bounding box of the detected camera. In the second (classification), the
target is the category of camera.

Once I have locations of cameras from a sample of intersections, the
target would be presence / density of cameras, with geographic
coordinates as the features (spatial regression, kriging, or other
spatial modeling methods).

In the first version of this project, I had two data sources, but I
recently added a third. Labeled images now come from Google Street View
(via Sheng, Yao, and Goel (2021)), the Objects365 dataset (Shao et al.
(2019)), and the Mapillary Vistas dataset (Neuhold et al. (2017)). There
are two types of images, full-sized images (from all 3 sources) and
cropped images (from Street View only). The cropped images were made in
Roboflow by cropping the full-sized images to their objects’ bounding
boxes. Because data came from different sources, standardizing and
unifying annotations was a major task in this project.

Full-sized images’ metadata are in YOLOv8 format, where each image has
an associated text file giving bounding box coordinates and labels.
Alongside this metadata are the folders of images. Cropped images are
arranged into folders by class, following the Pytorch `ImageFolder`
model.

#### Source

Sheng, Yao, and Goel (2021); Shao et al. (2019); and Neuhold et al.
(2017), with augmentation and annotation standardization done on the
Roboflow platform.

#### Size (jpg files)

- Full-size images: training, validation, and testing sets are 221MB,
  62MB, and 34MB, respectively
- Cropped images: training, validation, and testing sets are 4MB, 664kB,
  and 328kB, respectively

#### Dimensions after cleaning

Full-sized images and annotations:

| Data type | Split      | Files |
|:----------|:-----------|------:|
| Images    | Training   | 4,068 |
|           | Validation | 1,155 |
|           | Testing    |   617 |
| Labels    | Training   | 4,068 |
|           | Validation | 1,155 |
|           | Testing    |   617 |

Cropped images:

| Class    | Split      | Files |
|:---------|:-----------|------:|
| Directed | Training   |   517 |
|          | Validation |    75 |
|          | Testing    |    47 |
| Globe    | Training   |   471 |
|          | Validation |    84 |
|          | Testing    |    32 |

#### Time period

N/A

#### Data dictionary

In the YOLOv8 format, each image has a text file of one or more
annotations with no headings. One row = one marked camera

| Column number | Data type | Definition                              | Values                                           | Use                                   |
|---------------|-----------|-----------------------------------------|--------------------------------------------------|---------------------------------------|
| 1             | Int       | Class                                   | 0, 1, 2, 3 (for detection, I’ve removed classes) | Used to create classification folders |
| 2             | Float     | x-coordinate of the bounding box center | 0-1 scaled relative to image width               | Detection target                      |
| 3             | Float     | y-coordinate of the bounding box center | 0-1 scaled relative to image height              | Detection target                      |
| 4             | Float     | Bounding box width                      | 0-1 scaled relative to image width               | Detection target                      |
| 5             | Float     | Bounding box height                     | 0-1 scaled relative to image height              | Detection target                      |

## EDA

See notebook: [../src/eda_v2.ipynb](../src/eda_v2.ipynb)

## Model training and results

After a lot of trial and error and testing out a few different
frameworks, I settled on using Ultralytics YOLOv8 (Jocher, Chaurasia,
and Qiu (2023)). The YOLO family of models have been developed over
several years, and v8 is its latest iteration. These models are
relatively fast, dividing images into grids and estimating
classifications by region within those grids, avoiding more costly
anchor box calculations. The v8 package comes with trainer, tuner, and
predictor modules built in; has weights for a variety of computer vision
tasks; and has a few specialized variations on the YOLO algorithm. I
chose to use the package’s base YOLO model (for both detection and
classification) and the RT-DETR model, a new, transformer-based model
from Baidu (only available for detection). These models are pretrained
on benchmark datasets such as COCO, so I utilized the weights they
already had but fine-tuned them to my datasets; specifically,
surveillance cameras are not a class in the major benchmark datasets, so
I needed to adjust the models from being trained to detect 80 classes to
detecting just one. See details on the original YOLO implementation in
Redmon et al. (2016), and the RT-DETR model in Lv et al. (2023).

Most of the project was done in Python, with some bash scripting to help
with getting and processing data. The main Python packages I used were:

- Pytorch with CUDA
- PIL (for handling images)
- openCV (for handling images)
- ultralytics (models and YOLOv8 ecosystem of modules)
- albumentations (augmentations used by ultralytics)
- roboflow (interfacing with Roboflow platform)
- wandb (uploading artifacts to Weights & Biases)
- sahi (tiling for inference & data cleaning utilities)

In addition to those packages, other tools I used were:

- A Lenovo ThinkPad running Ubuntu 22.04 with 16 cores and 4GB GPU
  (development and tinkering with models before running on Paperspace)
- Roboflow (data management and augmentation)
- Weights & Biases (tracking runs & storing training artifacts)
- Paperspace (virtual machine with 8 CPUs, 16GB GPU, paid hourly)
- conda (environment management between workspaces)
- Hugging Face (hosting app)
- Voila (converting demonstration notebook to an app)
- Docker (containerization of the app)

For both the detection and classification tasks, I used 70/20/10 splits
for training, validation, and testing. I also created some variations on
the data. One issue I found was that the images from Mapillary are about
twice as big as the other two sources; cameras are already generally
very small relative to their images, and this variation made it even
more difficult. To fix this, I uploaded the datasets to Roboflow
separately, then created 2x2 tiles of the Mapillary images before
combining all the sources back together. This made the images come out
to similar dimensions. I also created a version of the dataset where all
images were cut into 3x3 tiles, again addressing the issue of detecting
small objects.

Altogether I trained 8 types of models for detection: for each
architecture (YOLO and RT-DETR), I trained a base model, a model with
all but the last few layers frozen, a model trained on tiled images, and
a model using both freezing and tiling. From there, I chose candidates
to continue tuning and training:

- A YOLO model with medium-sized weights
- YOLO with tiling
- RT-DETR with freezing

I found that the YOLO models are much faster to train—15 epochs can take
less than 30 minutes—but are less accurate and fail to detect some
cameras. Using the tiled dataset improved recall greatly, but doesn’t
necessarily transfer well to full-sized images. RT-DETR takes about 4
times as long, but is much more accurate. The main metric I used for
comparison was mAP 50, though I also paid attention to recall as a
metric that would be important in practice.

<figure>
<figcaption>
Training results, YOLO & RT-DETR models
</figcaption>
<img src="./imgs/training_results_line.png" />
</figure>

I then tuned the 2 YOLO models using Ultralytics’ tuning module, which
doesn’t seem to be fully implemented yet for RT-DETR. I also used the
model that had trained on tiled images to tune on full-sized images to
see how well it might transfer. However, tuning gave me pretty
lackluster results that sometimes performed worse than the untuned
models. This could be improved with more iterations and a more
methodical approach to tuning; Weights & Biases has tools for tuning
that use Bayesian probability for choosing hyperparameter values and
would work on both architectures.

<figure>
<figcaption>
Tuning results, YOLO variations only
</figcaption>
<img src="./imgs/tuning_results_box.png" />
</figure>

Because YOLO is a relatively small model and the cropped images for
classification are often no more than 20 pixels on either size, I was
able to do all the training and tuning for classification on my laptop’s
GPU. There are two classes available (globe camera or directed camera),
and these are only labeled on the small subset of images from Street
View, yet the accuracy was quite good even with a small sample size. The
YOLO classifier with medium weights achieves about 96% validation
accuracy after training for 25 epochs, and training and validation takes
just over a minute. However, many of the images are too small to meet
YOLO’s minimum pixel size, shrinking the sample size even further.

<figure>
<figcaption>
Confusion matrix, YOLO medium classifier, validation set
</figcaption>
<img src="./imgs/confusion_matrix.png" />
</figure>

## Demonstration

I used 2 of the best models to build a demonstration, YOLO trained on
tiled images and RT-DETR trained on tiled images with freezing. I used
Voila to turn a Jupyter notebook into an interactive app, with
JupyterLab widgets for controls. The app takes the text of a location,
such as a street intersection, plus parameters for the image size,
heading (angle), pitch (tilt), and field of view (zoom), and passes
these to the Google Street View static API. This returns an image, which
is then converted from bytes to a PIL Image object. The app then uses
both models to detect surveillance cameras in the image, and plot the
image with predicted bounding boxes overlaid; all three images are
placed side by side for comparison. I chose these two models to showcase
the differences in the architectures: the YOLO model runs quickly but,
in my testing, misses some cameras; the RT-DETR model takes a few
seconds to run, but catches more cameras. As it stands now, the RT-DETR
model is probably too heavy and slow to be useful on a low-RAM machine
or mobile device, or to make predictions in real time on video. I built
this in a Docker container, then deployed it to Hugging Face Spaces,
where it’s publicly available.

## Conclusion

Despite having many moving parts that needed to work together, I think
this project was successful. I built models that can detect surveillance
cameras in Street View images on demand and with a fair amount of
accuracy. The main shortcoming here is in tuning: I appreciate having
all the Ultralytics modules work together in one ecosystem, but their
tuner was inadequate for getting better performance of these models with
my limited equipment. I’d like to continue tuning, but switch most
likely to Weights & Biases. The models would likely perform better if
trained for longer—common benchmarks for comparing models often use 300
epochs or more.

I began dabbling in using sliced images for inference with SAHI (Akyon,
Onur Altinuc, and Temizel (2022)); it was from reading the research
behind that that I decided to tile my images, which turned out to work
quite well. One drawback is that if this is done in the app, it might
add more lag time to models that are already not as fast as I’d like.
However, there are frameworks available for optimizing models to perform
inference on smaller CPUs, which I haven’t done.

Even though this is a decent sized dataset with the addition of the
Mapillary images, it could still benefit from being larger. So far,
though, these are the only research datasets I’ve found that have
surveillance cameras labeled, and only Sheng, Yao, and Goel (2021) had
them labeled with different classes. I might go through the images I
already have and use the classification model to add class labels to the
Mapillary and Objects365 images, as well as add labels for types of
cameras more recently deployed, such as automated license plate readers.
I also might use AI labeling assistants to label cameras in other
streetscape datasets to create a larger training set.

Finally, this is just a first step toward using deep learning to study
the landscape of surveillance in Baltimore. The next major piece would
be to sample street intersections, detect cameras and deduplicate the
detections, and use the attached locations for spatial analysis.
Mapillary already has object detections labeled on their mapping
platform, including surveillance cameras, so this might just mean using
their location data and updating or supplimenting it with predictions
from these models.

While it is inherently reactionary to be chasing down the state’s
surveillance tools after they’ve been put up, I do feel there is a place
in surveillance studies and movements for police accountability to
implement open source data science developed by community members.

## References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-A.O.T2022" class="csl-entry">

Akyon, Fatih Cagatay, Sinan Onur Altinuc, and Alptekin Temizel. 2022.
“Slicing Aided Hyper Inference and Fine-Tuning for Small Object
Detection.” In *2022 IEEE International Conference on Image Processing
(ICIP)*, 966–70. <https://doi.org/10.1109/ICIP46576.2022.9897990>.

</div>

<div id="ref-Browne2015" class="csl-entry">

Browne, Simone. 2015. *Dark Matters: On the Surveillance of Blackness*.
Durham, NC: Duke University Press.

</div>

<div id="ref-J.C.Q2023" class="csl-entry">

Jocher, Glenn, Ayush Chaurasia, and Jing Qiu. 2023. “YOLO by
Ultralytics.” <https://github.com/ultralytics/ultralytics>.

</div>

<div id="ref-L.Z.X+2023" class="csl-entry">

Lv, Wenyu, Yian Zhao, Shangliang Xu, Jinman Wei, Guanzhong Wang, Cheng
Cui, Yuning Du, Qingqing Dang, and Yi Liu. 2023. “DETRs Beat YOLOs on
<span class="nocase">Real-time Object Detection</span>.” arXiv.
<https://doi.org/10.48550/arXiv.2304.08069>.

</div>

<div id="ref-N.O.R+2017" class="csl-entry">

Neuhold, Gerhard, Tobias Ollmann, Samuel Rota Bulo, and Peter
Kontschieder. 2017. “The Mapillary Vistas Dataset for Semantic
Understanding of Street Scenes.” In *Proceedings of the IEEE
International Conference on Computer Vision*, 4990–99.
<https://openaccess.thecvf.com/content_iccv_2017/html/Neuhold_The_Mapillary_Vistas_ICCV_2017_paper.html>.

</div>

<div id="ref-R.D.G+2016" class="csl-entry">

Redmon, Joseph, Santosh Divvala, Ross Girshick, and Ali Farhadi. 2016.
“You Only Look Once: Unified, Real-Time Object Detection.” arXiv.
<https://doi.org/10.48550/arXiv.1506.02640>.

</div>

<div id="ref-S.L.Z+2019" class="csl-entry">

Shao, Shuai, Zeming Li, Tianyuan Zhang, Chao Peng, Gang Yu, Xiangyu
Zhang, Jing Li, and Jian Sun. 2019. “Objects365: A Large-Scale,
High-Quality Dataset for Object Detection.” In *2019 IEEE/CVF
International Conference on Computer Vision (ICCV)*, 8429–38.
<https://doi.org/10.1109/ICCV.2019.00852>.

</div>

<div id="ref-S.Y.G2021" class="csl-entry">

Sheng, Hao, Keniel Yao, and Sharad Goel. 2021. “Surveilling
Surveillance: Estimating the Prevalence of Surveillance Cameras with
Street View Data.” In *Proceedings of the 2021 AAAI/ACM Conference on
AI, Ethics, and Society*, 221–30. AIES ’21. New York, NY, USA:
Association for Computing Machinery.
<https://doi.org/10.1145/3461702.3462525>.

</div>

<div id="ref-T.C.L+2021a" class="csl-entry">

Turtiainen, Hannu, Andrei Costin, Tuomo Lahtinen, Lauri Sintonen, and
Timo Hamalainen. 2021. “Towards Large-Scale, Automated, Accurate
Detection of CCTV Camera Objects Using Computer Vision. Applications and
Implications for Privacy, Safety, and Cybersecurity. (Preprint).” arXiv.
<http://arxiv.org/abs/2006.03870>.

</div>

<div id="ref-detectron" class="csl-entry">

Wu, Yuxin, Alexander Kirillov, Francisco Massa, Wan-Yen Lo, and Ross
Girshick. 2019. “Detectron2.”
<https://github.com/facebookresearch/detectron2>.

</div>

</div>
