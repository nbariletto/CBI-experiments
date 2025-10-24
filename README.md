# CBI-experiments
This repo collects reproducibility code for "Conformalized Bayesian Inference, with Applications to Random Partition Models."


The repo is organized as follows: experiments are grouped depending on whether they appear in the main body or the supplementary material of the article. In each subfolder corresponding to an experiment, you can find a single .py script to reproduce that experiment. If data is required for the experiment, that data is also in the folder.

Note: for the colorectal cancer spatial transcriptomic experiment, we have relied on MCMC output produced by Yunshan Duan and collaborators in their paper "Spatially aligned random partition models on spatially resolved transcriptomics data." This MCMC output is in the corresponding experiment subfolder and is taken as input by the .py script in that same subfolder. You consult the original paper [here](https://www.biorxiv.org/content/10.1101/2025.04.16.649218v1).
