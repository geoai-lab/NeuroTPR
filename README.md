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

* "HarveyTweet" folder: This folder contains the Harvey2017 dataset with 1,000 human-annotated tweets.
* "Model" folder: This folder contains the Python source codes to use the trained NeuroTPR model or retrain NeuroTPR for toponym recognition.
* "WikiDataHelper" folder: This folder contains the Python source codes to build up an annotated dataset from Wikipedia for training NeuroTPR.
* "training_data" folder: This folder contains three training data sets (Wikipedia3000, WNUT2017, and 50 optional tweets from Hurricane Harvey) used for training NeuroTPR. Wikipedia3000 was automatically constructed from 3000 Wikipedia articles using our proposed workflow (more details can be found in the folder WikiDataHelper); WNUT2017 contains 599 tweets selected from [the original dataset](https://github.com/leondz/emerging_entities_17); and 50 optional tweets contain 50 crisis-related tweets from the Hurricane Harvey Twitter Dataset which contain door number addresses or street names.



### Project dependencies:

* Python 3.6+ and a recent version of numpy
* Keras 2.3.0
* Tensorflow 1.8.0+
* Keras-contrib (https://github.com/keras-team/keras-contrib)
* Tensorflow Hub (https://www.tensorflow.org/hub)
* The rest should be installed alongside the four major libraries

### Retrain NeuroTPR using your own data

If you wish to perform re-training on your own dataset, you have to add POS features to your own annoated dataset in CoNLL2003 format.
You can use the following python codes to add POS features via NLTK tool.

```bash
    python3 Model/add_lin_features.py
```

Train NeuroTPR(See codes for further modification to fit your own need). You may need to:
* Set up the file path to load word embeddings, training data;
* Set up the file path to save the trained model;
* Tune the key hyper-parameters of the NeuroTPR

```bash
    python3 Model/train.py
 ```

### Use the NeuroTPR model for toponym recognition

The following python codes provide a example to use the trained model to recognize toponyms from texts.

Input: Tokenized texts saved in CoNLL2003 format file

```bash
    python3 Model/geoparsing.py
 ```
Model output: toponym-name1,,statr-index,,end-index||toponym-name2,,statr-index,,end-index||...


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
