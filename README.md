# Chexpert Student-Teacher Framework

Images on Chest X-Ray have been trained using the implementation provided [here](https://github.com/abkds/DeepLearningImplementations/tree/master/DenseNet).

You can read more about the dataset in this [paper](https://arxiv.org/abs/1901.07031).

### Data
The chexpert data was uploaded to floydhub and from there it is being used for experiments by running these files. Here is the [link](https://www.floydhub.com/abkds/datasets/chexpert). The process uses the validation data set while doing the optimization which has also been uploaded [here](https://www.floydhub.com/abkds/datasets/chexpert_validation)

### Run on FloydHub
To run the networks on Floydhub you can use the following command on the files from terminal. 

```
floyd run --gpu2 --env keras --data abkds/datasets/atelectasis/1:atelectasis --data abkds/datasets/chexpert_validation/1:chexpert_validation 'python atelectasis.py'
```

### Weights

The weights are around `100 Mb` so it was not possible to upload them here or neither in the repo for GUI. A drive link for that has been given from where it can be downloaded and combined with the GUI repo to run the repository.



