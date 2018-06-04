# deepNF
This repository contains Python scripts for "deepNF: Deep network fusion for protein function prediction" by V. Gligorijevic, M. Barot and R. Bonneau.


# Citing
```
@article {Gligorijevic2017,
	author = {Gligorijevi{\'c}, Vladimir and Barot, Meet and Bonneau, Richard},
	title = {deepNF: Deep network fusion for protein function prediction},
	year = {2018},
	doi = {10.1093/bioinformatics/bty440},
    pages = {bty440},
	publisher = {Oxford},
	URL = {http://dx.doi.org/10.1093/bioinformatics/bty440},
	journal = {Bioinformatics}
}
```
## Usage

To run *deepNF* run the following command from the project directory:
```
python2.7 main.py example_params.txt
```
## Dependencies

*deepNF* is tested to work under Python 2.7.

The required dependencies for *deepNF* are [Keras](https://keras.io/), [TensorFlow](https://www.tensorflow.org/), [Numpy](http://www.numpy.org/), and [scikit-learn](http://scikit-learn.org/).

## Data

Data (PPMI matrices for human and yeast STRING networks as well as protein annotations) used for producing figures in the paper can be downloaded from:

https://drive.google.com/drive/folders/1iDz_kj41YohUq5nukI0GcBAjXsyI7xEW?usp=sharing
