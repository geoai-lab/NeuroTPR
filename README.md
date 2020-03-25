# NeuroTPR
This is Neuro-net ToPonym Recognition model for extracting locations from social media messages using deep Recurrent Neural Network. 

The goal of this model is to improve the toponym (location name entity) recognition performance from social media messages that have various
language irregularities, and fine-grained locations such as streets, natural landscapes, facilities, and townships.

The full paper is available at: [NeuroTPR: A Neuro-net ToPonym Recognition Model for Extracting Locations from Social Media Messages](https://geoai.geog.buffalo.edu/publications/)

### NeuroTPR architecture

<p align="center">
<img align="center" src="Model_on_paper.png" width="600" />
</p>

### Training Dataset

* 599 tweets (have at least one location entity) selected from [WNUT-2017](https://github.com/leondz/emerging_entities_17)

* A large geo-annotated dataset based on Wikipedia articles (See dataset construction details in WikiDataHelper)

* Optional 50 crisis-related tweets from Hurricane Harvey Twitter Dataset: which contain door number addresses or street names

* All datasets are converted into the same format as WNUT-2017

### NeuroTPR training

Dependencies:

* Python 3.6+ and a recent version of numpy
* Keras 2.3.0
* Tensorflow 1.8.0+
* Keras-contrib (https://github.com/keras-team/keras-contrib)
* Tensorflow Hub (https://www.tensorflow.org/hub)
* The rest should be installed alongside the four major libraries


Optional: Add POS and NER features to your own annoated dataset

```bash
    python3 Model/add_lin_features.py
```

Train NeuroTPR (See codes for further modification to fit your own need)

```bash
    python3 Model/train.py
 ```

### NeuroTPR prediction on batch

```bash
    python3 Model/geoparsing.py
 ```

See as the file example of predicting model's input and output

### Performance evaluation

* HarveyTweet: 1,000 human annotated tweets derived from a large Twitter dataset collected during Hurricane Harvey
* GeoCorproa:  1,689 tweets from the study of [GeoCorpora Project](https://github.com/geovista/GeoCorpora)
* Ju2016: 5,000 short human generated sentecnes maintaned at [EUPEG Project](https://github.com/geoai-lab/EUPEG/tree/master/corpora/Ju2016)

We test NeuroTPR using the benchmarking platform [EUPEG](https://github.com/geoai-lab/EUPEG). The performance of NeuroTPR on three dataset is presented in the table below:

|   Corpora   |  Precision |  Recall   |   F_score  |
|-------------|:----------:|----------:|-----------:|
| HarveyTweet |    0.755   |   0.695   |	0.724	|
|  GeoCorpora |    0.817   |   0.745   |	0.779	|
|    Ju2016   | 	 -	   |   0.636   |	  - 	|
