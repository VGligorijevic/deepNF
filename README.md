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
python main.py
```
To see the list of options:
```
python main.py --help
```

To compute network emgeddings only use *net_embedding.py* script. Input file
format: edgelist (i, j, w\_ij)

For a single network:


```
python net_embedding.py --model_type ae --nets example_net_1.txt
```

For multiple networks:

```
python net_embedding.py --model_type mda --nets example_net_1.txt example_net_2.txt
```



## Dependencies

*deepNF* is tested to work under Python 3.6.

The required dependencies for *deepNF* are [Keras](https://keras.io/), [TensorFlow](https://www.tensorflow.org/), [Numpy](http://www.numpy.org/), [NetworkX](https://networkx.github.io/) and [scikit-learn](http://scikit-learn.org/).

## Data

Data (PPMI matrices for human and yeast STRING networks as well as protein annotations) used for producing figures in the paper can be downloaded from:

https://users.flatironinstitute.org/vgligorijevic/public_www/deepNF_data/
