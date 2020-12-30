# Divergence Regulated Encoder Network:
**Divergence Regulated Encoder Network For Joint Dimensionality Reduction And Classification**

_Joshua Peeples, Sarah Walker, Connor McCurley, Alina Zare, and James Keller_

Note: If this code is used, cite it: Joshua Peeples, Sarah Walker, Connor McCurley, Alina Zare, & James Keller. 
(Date). GatorSense/DREN: Initial Release (Version v1.0). 
Zendo. https://doi.org/10.5281/zenodo.4404604 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4404604.svg)](https://doi.org/10.5281/zenodo.4404604)

[[`arXiv`](link to ArXiv paper will be posted soon)]

[[`BibTeX`](#CitingHist)]


In this repository, we provide the paper and code for Divergence Regulated Encoder Network (DREN) models from "Divergence Regulated Encoder Network For Joint Dimensionality Reduction And Classification"

## Installation Prerequisites

This code uses python, pytorch, and barbar. 
Please use [[`Pytorch's website`](https://pytorch.org/get-started/locally/)] to download necessary packages.
Barbar is used to show the progress of model. Please follow the instructions [[`here`](https://github.com/yusugomori/barbar)]
to download the module.

## Demo

Run `demo.py` in Python IDE (e.g., Spyder) or command line. To evaluate performance,
run `View_Results.py` (if results are saved out).

## Main Functions

The Divergence Regulated Encoder Network (DREN) runs using the following functions. 

1. Intialize model  

```model, input_size = intialize_model(**Parameters)```

2. Prepare dataset(s) for model

 ```dataloaders_dict = Prepare_Dataloaders(**Parameters)```

3. Train model 

```train_dict = train_model(**Parameters)```

4. Test model

```test_dict = test_model(**Parameters)```


## Parameters
The parameters can be set in the following script:

```Demo_Parameters.py```

## Inventory

```
https://github.com/GatorSense/DREN

└── root dir
    ├── demo.py   //Run this. Main demo file.
    ├── Demo_Parameters.py // Parameters file for demo.
    ├── Prepare_Data.py  // Load data for demo file.
    ├── Texture_Information.py // Class names and directories for datasets.
    ├── View_Results.py // Run this after demo to view saved results.
    ├── View_Results_Parameters.py // Parameters file for results.
    ├── knn_experiment.py // Trains and tests a KNN with the embeddings produced by the model and embeddings produced through t-SNE
    ├── Out_of_Sample.py // Produces an embedding with out of sample points by learning the manifold of the original embedding
    ├── papers  // Related publications.
    │   ├── readme.md //Information about paper
    └── Utils  //utility functions
        ├── Compute_FDR.py  // Compute Fisher Discriminant Ratio for features.
        ├── Confusion_mats.py  // Generate confusion matrices.
        ├── Embedding_Model.py  // Generates model with an encoder following the final layer 
        ├── Generate_Embedding_Vid.py  // Generates a video showing how the embedding of the model changes with each epoch
        ├── Generate_Histogram_Vid.py  // Generates a video showing how the histogram layer varies with each epoch
        ├── Generate_Learning_Curves.py  // Generates the learning curves for the model
        ├── Generate_TSNE_visual.py  // Generate TSNE visualization for features.
        ├── Histogram_Model.py  // Generate HistRes_B models.
        ├── Network_functions.py  // Contains functions to initialize, train, and test model. 
        ├── Plot_Decision_Boundary.py // Plots the decision boundary found by the model
        ├── RBFHistogramPooling.py  // Create histogram layer. 
        ├── Save_Results.py  // Save results from demo script.
        ├── TSNE_Loss.py  // Includes functions to compute the embedding loss found by t-SNE methods
     
```

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

This product is Copyright (c) 2021 J. Peeples, S. Walker, C. McCurley, A. Zare, & J. Keller. All rights reserved.

## <a name="CitingHist"></a>Citing Divergence Regulated Encoder Network (DREN)

If you use the Divergence Regulated Encoder Network (DREN) code, please cite the following reference using the following entry.

**Plain Text:**

Peeples, J., Walker, S., McCurley, C., Zare, A., & Keller, J. (2021). Divergence Regulated Encoder Network For Joint Dimensionality Reduction And Classification. arXiv preprint (will be posted soon).

**BibTex:**
```
@article{peeples2021divergence,
  title={Divergence Regulated Encoder Network For Joint Dimensionality Reduction And Classification},
  author={Peeples, Joshua and Walker, Sarah and McCurley, Connor and Zare, Alina and Keller, James},
  journal={arXiv will be posted soon},
  year={2021}
}
```

