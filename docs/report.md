# Proposal: Re-surveilling surveillance
Camille Seaberry

Prepared for UMBC Data Science Master Degree Capstone with Dr. Chaojie
Wang

- https://github.com/camille-s

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

For this project, I have **two major goals:**

1.  Improving upon the models I used, including introduction of more
    predefined models (adding YOLO, among others), finer tuning of
    classification of camera type (including possibly adding automated
    license plate readers), more concerted sampling of intersections in
    Baltimore, and updated images
2.  Mapping of those locations to study the landscape of that layer of
    surveillance. In the longer term, I would like to do some amount of
    spatial analysis overlaying camera locations and socio-economic /
    demographic data to understand any patterns in this landscape and
    the potential burdens of surveillance on marginalized communities in
    Baltimore.

The first of these is the focus of my capstone, while the second is a
longer-term possibility.

Therefore my **major research questions are:**

1.  How accurately can deep learning models detect surveillance cameras
    in street images?
2.  How accurately can deep learning models classify types of
    surveillance cameras?
3.  What spatial patterns exist in where surveillance cameras are
    located?

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

### Images

There are two sets of images, full-sized images and cropped images. The
cropped images were made in Roboflow by cropping the full-sized images
to their objects’ bounding boxes. Full-sized images’ metadata are in
COCO JSON format, with metadata about the date of upload to Roboflow,
licenses, categories, images, and annotations. Of these, I am using the
data on images and annotations. Alongside this metadata are the folders
of images. Cropped images are arranged into folders by class, following
the Pytorch `ImageFolder` model.

#### Source

Sheng, Yao, and Goel (2021) and Shao et al. (2019), with metadata
standardized on the Roboflow platform.

#### Size (jpg files)

- Full-size images: training, validation, and testing sets are 168MB,
  18MB, and 8.5MB, respectively
- Cropped images: training, validation, and testing sets are 6MB, 596kB,
  and 296kB, respectively

#### Dimensions after cleaning

Full-sized images and annotations:

| Data type   | Split      | Rows  | Columns |
|-------------|------------|-------|---------|
| Images      | Training   | 2,580 | 7       |
|             | Validation | 244   | 7       |
|             | Testing    | 121   | 7       |
| Annotations | Training   | 4,654 | 7       |
|             | Validation | 474   | 7       |
|             | Testing    | 236   | 7       |

#### Time period

N/A

#### Data dictionary

Note that because annotations are in a standardized format (COCO JSON),
some columns can be disregarded for analysis but I am describing them
here for completeness.

##### Images

One row = one image

| Name          | Data type | Definition                | Values        | Use                   |
|---------------|-----------|---------------------------|---------------|-----------------------|
| id            | Int       | Unique numeric image ID   |               | Join with annotations |
| license       | Int       | License type              | 1 (CC BY 4.0) | Disregard             |
| file_name     | String    | Image file name           |               | Path to read images   |
| height        | Int       | Full image height         | 640 (uniform) |                       |
| width         | Int       | Full image width          | 640 (uniform) |                       |
| data_captured | Datetime  | Date uploaded to Roboflow | 2023-10-22    |                       |
| extra         | String    | Additional notes          | empty         | Disregard             |

##### Annotations

One row = one marked camera

| Name         | Data type      | Definition                                                                               | Values                                                   | Use                                         |
|--------------|----------------|------------------------------------------------------------------------------------------|----------------------------------------------------------|---------------------------------------------|
| id           | Int            | Unique numeric annotation ID                                                             |                                                          |                                             |
| image_id     | Int            | Unique numeric image ID of corresponding image                                           |                                                          | Join with images                            |
| category_id  | Int            | Numeric category of camera type                                                          | 1: directed, 2: globe, 3: no classification (Objects365) | Target for classification of cropped images |
| bbox         | List of 4 ints | Bounding box of annotation in COCO coordinate format (center-x, center-y, width, height) |                                                          | Target for object detection                 |
| area         | Float          | Area of bounding box                                                                     |                                                          | May be used as a feature in classification  |
| segmentation | List of ints   | Coordinates of segmentation points                                                       | Empty                                                    | Disregard                                   |
| iscrows      | Int            | Flag for presence of “crowd” in image                                                    | Binary                                                   | Disregard                                   |

### Street network

If I get models working well on detecting and classifying cameras in
these older images, I’d like to create a set of new up-to-date images
from Google Street View. To do this, following the methodology in Sheng,
Yao, and Goel (2021), I will sample intersections from a network of
Baltimore streets to get coordinates at which to batch download Street
View images using Google’s API. After cleaning, the result is a
geopandas GeoDataFrame of coordinates of street intersections extracted
from the network.

#### Source

Road network geography data comes from OpenStreetMap via the OSMnx
package (Boeing (2017)).

#### Size

- Full network, geopackage file: 36MB (not used for analysis)
- Intersection coordinates, geopackage file: 3.8MB

#### Dimensions after cleaning

24,274 rows x 4 columns

#### Time period

N/A

#### Data dictionary

One row = one intersection node

| Name     | Data type | Definition                         | Values | Use                           |
|----------|-----------|------------------------------------|--------|-------------------------------|
| id       | Int       | ID from the OpenStreetMap database |        |                               |
| lon      | Float     | Longitude coordinate               | ~-76   | Use for Street View API calls |
| lat      | Float     | Latitude coordinate                | ~39    | Use for Street View API calls |
| geometry | Point     | GeoSeries of coordinates           |        | Use for spatial analysis      |

## EDA

See notebook: [../src/eda_v2.ipynb](../src/eda_v2.ipynb)

## References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-Boeing2017" class="csl-entry">

Boeing, Geoff. 2017. “OSMnx: New Methods for Acquiring, Constructing,
Analyzing, and Visualizing Complex Street Networks.” *Computers,
Environment and Urban Systems* 65 (September): 126–39.
<https://doi.org/10.1016/j.compenvurbsys.2017.05.004>.

</div>

<div id="ref-Browne2015" class="csl-entry">

Browne, Simone. 2015. *Dark Matters: On the Surveillance of Blackness*.
Durham, NC: Duke University Press.

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