# NeuroTPR

### Overall description

NeuroTPR is a toponym recognition model designed for extracting locations from social media messages. It is based on a general Bidirectional Long Short-Term Memory network (BiLSTM) with a number of additional features, such as double layers of character embeddings, GloVe word embeddings, and contextualized word embeddings ELMo.

The goal of this model is to improve the accuracy of toponym recognition from social media messages that have various
language irregularities, such as informal sentence structures, inconsistent upper and lower cases (e.g., “there is a HUGE fire near camino and springbrook rd”), name abbreviations (e.g., “bsu” for “Boise State University”), and misspellings. Particularly, NeuroTPR is designed to extract fine-grained locations such as streets, natural features, facilities, point of interest (POIs), and administrative units. We tested NeuroTPR in the application context of disaster response based on a dataset of tweets from Hurricane Harvey in 2017.

More details can be found in our paper: [Wang, J., Hu, Y., & Joseph, K. (2020): NeuroTPR: A Neuro-net ToPonym Recognition model for extracting locations from social media messages. Transactions in GIS, accepted.](https://geoai.geog.buffalo.edu/publications/)

<p align="center">
<img align="center" src="model_structure.png" width="600" />
<br />
Figure 1. The overall architecture of NeuroTPR
</p>

### Repository organization

* HarveyTweet: Harvey2017 dataset
* Model folder: Python source codes to retrain NeuroTPR and use the trained model for toponym recognition 
* WikiDataHelper: Python source codes to build up an annotated dataset from Wikipedia for training NeuroTPR
* training_data: Three training data sets used in the default model training


### Training Dataset

* 599 tweets (each has at least one location entity) selected from [WNUT-2017](https://github.com/leondz/emerging_entities_17)

* A large geo-annotated dataset based on Wikipedia articles (See dataset construction details in WikiDataHelper)

* Optional 50 crisis-related tweets from Hurricane Harvey Twitter Dataset: which contain door number addresses or street names


### Retrain NeuroTPR using your own data?

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

### Use the NeuroTPR model for toponym recognition

```bash
    python3 Model/geoparsing.py
 ```

example folder provides the file example of input and output for NeuroTPR prediction function


### Performance evaluation

* HarveyTweet: 1,000 human annotated tweets derived from a large Twitter dataset collected during Hurricane Harvey
* GeoCorproa:  1,689 tweets from the study of [GeoCorpora Project](https://github.com/geovista/GeoCorpora)
* Ju2016: 5,000 short human generated sentecnes maintaned at [EUPEG Project](https://github.com/geoai-lab/EUPEG/tree/master/corpora/Ju2016)

We test NeuroTPR using the benchmarking platform [EUPEG](https://github.com/geoai-lab/EUPEG). The performance of NeuroTPR on three dataset is presented in the table below:

|   Corpora   |  Precision |  Recall   |   F_score  |
|-------------|:----------:|----------:|-----------:|
| HarveyTweet |    0.787   |   0.678   |	0.728	|
|  GeoCorpora |    0.800   |   0.761   |	0.780	|
|    Ju2016   | 	 -	   |   0.821   |	  - 	|
