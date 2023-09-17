# Proposal

# Re-surveilling surveillance, Camille Seaberry

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
image, its location is known. For my capstone, I have two major
goals: 1) improving upon the models I used, including introduction of
more predefined models (adding YOLO, among others), finer tuning of
classification of camera type (including possibly adding automated
license plate readers), more concerted sampling of intersections in
Baltimore, and updated images; and 2) mapping of those locations to
study the landscape of that layer of surveillance. If possible, my third
goal would be some amount of spatial analysis overlaying camera
locations and socio-economic / demographic data to understand any
patterns in this landscape and the potential burdens of surveillance on
marginalized communities in Baltimore.

## Data & EDA

See notebook: [./00_eda.ipynb](./00_eda.ipynb)

## References

<div id="refs" class="references csl-bib-body hanging-indent">

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
