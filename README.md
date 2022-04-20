# golden-gaze: a Pipeline for Reducing Health Misinformation in Search

This is the code repository for the paper: 
Dake Zhang, Amir Vakili Tahami, Mustafa Abualsaud, and Mark D. Smucker. 
"Learning Trustworthy Web Sources to Derive Correct Answers and Reduce Health Misinformation in Search." 
In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval. 2022.
[Link to be updated]

## Data

This pipeline was evaluated on the [TREC Health Misinformation Track](https://trec-health-misinfo.github.io/).
The topics and qrels used in our paper are hosted on the [TREC](https://trec.nist.gov/) website.
We also used two web collections: [ClueWeb12-B13](https://lemurproject.org/clueweb12/) and en.noclean version of [C4](https://www.tensorflow.org/datasets/catalog/c4).
`data/WH_topics.csv` contains 90 topics (45 helpful topics and 45 topics) sampled from the `external_data/Truth.txt` originated from Ryen W White and Ahmed Hassan, where overlapping topics with 2019 or 2021 topics were removed. 

Download those file listed below and put them in the folder `data/raw_data/`.

1. 2019 Topics `2019topics.xml`: https://trec.nist.gov/data/misinfo/2019topics.xml
2. 2019 Topics Efficacy Labels `2019topics_efficacy.txt`: https://trec.nist.gov/data/misinfo/2019topics_efficacy.txt
3. 2019 qrels `2019qrels_raw.txt`: https://trec.nist.gov/data/misinfo/2019qrels_raw.txt
4. 2021 Topics `misinfo-2021-topics.xml`: https://trec.nist.gov/data/misinfo/misinfo-2021-topics.xml
5. 2021 qrels `qrels-35topics.txt`: located at `qrels/qrels-35topics.txt` within https://trec.nist.gov/data/misinfo/misinfo-resources-2021.tar.gz

## Environment
We use Python 3.8.10 with the following main packages.
```
beautifulsoup4==4.9.3
nltk==3.5
pyserini==0.13.0
pytorch-lightning==1.5.8
scikit-learn==0.24.2
torch==1.9.0
transformers==4.6.1
```


## Experiment
In this section, we provide a step-by-step guidance to reproduce the results reported in our paper.
We have provided all code in this repository.
Meanwhile, we have also provided intermediate outputs after each stage.
In other words, you can quickly experiment with any stage without the need to run previous stages to get the input data to that stage.

### Stage 1: Initial Retrieval

### Stage 2: Stance Detection

### Stage 3: Answer Prediction

### Stage 4: Reranking

### Stage 5: Evaluation


## Citation
Please cite the following paper if you use our code.
```
@inproceedings{zhang2022trustworthy,
  title={Learning Trustworthy Web Sources to Derive Correct Answers and Reduce Health Misinformation in Search},
  author={Zhang, Dake and Vakili Tahami, Amir and Abualsaud, Mustafa and Smucker, Mark D.},
  booktitle={Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  year={2022}
}
```

## Acknowledgement
This work was supported in part by the Natural Sciences and Engineering Research Council of Canada (RGPIN-04665-2020, RGPAS-00080-2020), in part by Mitacs, in part by Compute Canada, and in part by the University of Waterloo.

## Contact
If you have questions regarding this repository, 
please contact Dake using https://zhangdake.com.cn/#contact.
