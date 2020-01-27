# NeuroTPR
This is Neuro-net ToPonym Recognition model for extracting locations from social media messages using deep Recurrent Neural Network. 

The goal of this model is to improve the toponym recognition performance from social media messages that have various
language irregularities, and fine-grained locations such as streets, natural landscapes, facilities, and townships.

The full paper is available here:[NeuroTPR: A Neuro-net ToPonym Recognition Model for Extracting Locations from Social Media Messages]("https://geoai.geog.buffalo.edu")

### NeuroTPR architecture

<p align="center">
<img align="center" src="Model_on_paper.png" width="600" />
</p>

### Dataset

* [WNUT-2017](https://github.com/leondz/emerging_entities_17) as default training dataset

* A large geo-annotated dataset based on Wikipedia articles (See dataset construction details in WikiDataHelper)

* 50 place-related tweets selected from Hurricane Harvey Twitter Dataset


### NeuraTPR training

```bash
    python3 Model/train.py
 ```

### NeuraTPR prediction on batch

