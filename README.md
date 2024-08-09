# IDGC: Interpretable Deep Graph Clustering

This repository contains the implementation of **IDGC (Interpretable Deep Graph Clustering)**, a novel approach for interpretable and efficient graph clustering. Our work has been accepted for presentation at **ICPR 2024 (International Conference on Pattern Recognition)**.

## Overview

IDGC is a deep learning-based framework designed for clustering graph-structured data. It leverages graph neural networks (GNNs) combined with interpretable clustering techniques to provide meaningful insights into complex graph data.

## Requirements

Before running the code, ensure you have installed the necessary dependencies:

```bash
conda create -n idgc_env python=3.8
conda activate idgc_env

# Install PyTorch and PyTorch Geometric
pip install torch==1.8.0 torch-geometric==2.0.2

# Additional dependencies
conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl
pip install matplotlib
pip install pandas
pip install soyclustering
pip install rdkit
```

## Usage

To run IDGC, execute the following command:

```bash
python -m models.train
```

### Configuration

You can modify the configuration settings in `utils/Configures.py` according to your dataset and experimental setup.

## Citation

If you find this work useful in your research, please consider citing:

```
@inproceedings{your_paper_reference,
  title={Interpretable Deep Graph Clustering},
  author={Your Name and Others},
  booktitle={Proceedings of the 2024 International Conference on Pattern Recognition (ICPR)},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

We would like to thank the community for their valuable contributions and support. 
