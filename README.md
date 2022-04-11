# Divergence Regulated Encoder Network:
**Divergence Regulated Encoder Network For Joint Dimensionality Reduction And Classification**

_Joshua Peeples, Sarah Walker, Connor McCurley, Alina Zare, James Keller and Weihuang Xu_

Note: If this code is used, cite it: Joshua Peeples, Sarah Walker, Connor McCurley, Alina Zare, James Keller, & Weihuang Xu. 
(2020, December 30). GatorSense/DREN: Initial Release (Version v1.0). 
Zenodo. https://doi.org/10.5281/zenodo.4404604 
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4404604.svg)](https://doi.org/10.5281/zenodo.4404604)

[[`IEEE GRSL`](https://doi.org/10.1109/LGRS.2022.3156532)]

[[`arXiv`](https://arxiv.org/abs/2012.15764)]

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
    ├── Capture_Metrics.py // Save validation and test performance in spreadsheet.
    ├── Convergence_Analysis.py // Analyze convergence of models for each dataset.
    ├── Prepare_Data.py  // Load data for demo file.
    ├── Texture_Information.py // Class names and directories for datasets.
    ├── View_Results.py // Run this after demo to view saved results.
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
        ├── Plot_Accuracy.py // Plots the average and std of metrics for each model
        ├── Plot_Decision_Boundary.py // Plots the decision boundary found by the model.    
        ├── RBFHistogramPooling.py  // Create histogram layer. 
        ├── Save_Results.py  // Save results from demo script.
        ├── TSNE_Loss.py  // Includes functions to compute the embedding loss found by t-SNE methods
     
```

## License

This source code is licensed under the license found in the [`LICENSE`](LICENSE) file in the root directory of this source tree.

This product is Copyright (c) 2022 J. Peeples, S. Walker, C. McCurley, A. Zare, J. Keller, & W. Xu. All rights reserved.

## <a name="CitingHist"></a>Citing Divergence Regulated Encoder Network (DREN)

If you use the Divergence Regulated Encoder Network (DREN) code, please cite the following reference using the following entry.

**Plain Text:**

J. Peeples, S. Walker, C. Mccurley, A. Zare, J. Keller and W. Xu, "Divergence Regulated Encoder Network for Joint Dimensionality Reduction and Classification," in IEEE Geoscience and Remote Sensing Letters, vol. 19, pp. 1-5, 2022, Art no. 3511305, doi: 10.1109/LGRS.2022.3156532.

**BibTex:**
```
@ARTICLE{peeples2022divergence,
  author={Peeples, Joshua and Walker, Sarah and Mccurley, Connor and Zare, Alina and Keller, James and Xu, Weihuang},
  journal={IEEE Geoscience and Remote Sensing Letters}, 
  title={Divergence Regulated Encoder Network for Joint Dimensionality Reduction and Classification}, 
  year={2022},
  volume={19},
  number={},
  pages={1-5},
  doi={10.1109/LGRS.2022.3156532}
  }
```

