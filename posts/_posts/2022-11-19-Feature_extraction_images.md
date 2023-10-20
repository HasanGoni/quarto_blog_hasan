# Introduction

* This is a part of the series. [First](https://Hasangoni.github.io/2022/11/19/Explanatory_data_analysis.html), [Second](https://Hasangoni.github.io/2022/11/19/Feature_preprocessing_and_generation.html), [Third](https://Hasangoni.github.io/2022/11/19/Exploring_anonymized_data.html) and [Fourth](https://Hasangoni.github.io/2022/11/19/Feature_extraction_from_text.html) parts are connected in link. In this part I will try to write about EDA, when the data is image type

# Images Vector extraction

* Similar to text vector extraction, we can extract features from images
* Besides getting outputs from last layer, we also can get outputs from intermediate layers, we call them descriptors.
* Descriptors from later layers are better to solve task similar to one network was trained on.
* In contraray descriptors from earlier layers has some task independent features.
* Image net descriptors from last layer can be used for like car classification, but for medical image task earlier layers descriptos may be a better choice or train from scratch.
* Fine tuning is a good option to get better results.
  * e.g. Data Science game 2016. The task was to classify roofs into groups. Log Loss was the metric. Competitors has 8000 images. Fine tuning with image net helps to get better results.
* Augmentations:
  * Create different versions of same image, may be rotation, random crop, etc.
* Be careful with validation and don't overfit.