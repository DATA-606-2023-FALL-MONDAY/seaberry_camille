# Proposal: Re-surveilling surveillance
Camille Seaberry

Prepared for UMBC Data Science Master Degree Capstone with Dr. Chaojie
Wang

- https://github.com/camille-s

## Background

Police surveillance cameras in Baltimore—many of which are situated at
street intersections that make up a spatial network by which people move
through the city—form one of many layers of state surveillance imporsed
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

For my capstone, I have **two major goals:**

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

If I decide to do spatial analysis once I have locations of cameras from
a sample of intersections, the target would be presence / density of
cameras, with geographic coordinates as the features (spatial
regression, kriging, or other spatial modeling methods).

### Images

There are two sets of images, full-sized images and cropped images. The
cropped images were made in Roboflow by cropping the full-sized images
to their objects’ bounding boxes. Metadata are in COCO JSON format, with
metadata about the date of upload to Roboflow, licenses, categories,
images, and annotations. Of these, I am using the data on images and
annotations. Alongside this metadata are the folders of images.

#### Source

Sheng, Yao, and Goel (2021) and Shao et al. (2019), with metadata
standardized on the Roboflow platform.

#### Size (jpg files)

- Full-size images: training, testing, and validation sets are 495MB,
  28.6MB, and 53.2MB, respectively
- Cropped images: training, testing, and validation sets are 1.1MB,
  356kB, and 994kB, respectively

#### Dimensions after cleaning

- Annotations: 5,655 rows x 6 columns (2 indices)
- Image metadata: 3,557 rows x 6 columns (2 indices)

#### Time period

N/A

#### Data dictionary

##### Annotations

One row = one marked camera

| Name        | Data type      | Definition                                           | Values                                                                     | Use                                         |
|-------------|----------------|------------------------------------------------------|----------------------------------------------------------------------------|---------------------------------------------|
| type        | String (index) | Image type: full-size vs cropped image               | “full”, ‘crop’                                                             |                                             |
| id          | Int (index)    | Numeric ID marking the annotation within image types | 0-(number of rows by type)                                                 |                                             |
| image_id    | Int            | Numeric ID of corresponding image                    | 0-(number of images)                                                       |                                             |
| category_id | Int            | Numeric category of camera type                      | 1: directed; 2: globe; 3: no classification (Objects365)                   | Target for classification of cropped images |
| bbox        | List of 4 ints | Bounding box of camera within image                  | Coordinates of top-left corner w/r/t image, width & height of bounding box | Target for object detection                 |
| area        | Float          | Area of the bounding box                             |                                                                            |                                             |

##### Images

One row = one image

| Name          | Data type      | Definition                                      | Values                     | Use                                                               |
|---------------|----------------|-------------------------------------------------|----------------------------|-------------------------------------------------------------------|
| type          | String (index) | Image type: full-size vs cropped image          | “full”, “crop”             |                                                                   |
| id            | Int (index)    | Numeric ID marking the image within image types | 0-(number of rows by type) | After reading from this file, pixel data tensors will be features |
| file_name     | String         | File name to read image                         |                            |                                                                   |
| height        | Int            | Height of image                                 |                            |                                                                   |
| width         | Int            | Width of image                                  |                            |                                                                   |
| date_captured | Datetime       | Date of upload to Roboflow                      | Currently all May 2023     |                                                                   |

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

See notebook: [./eda.ipynb](./eda.ipynb)

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
