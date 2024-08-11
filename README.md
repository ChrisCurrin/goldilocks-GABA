# Goldilocks GABA: Network models incorporating chloride dynamics predict optimal strategies for terminating status epilepticus

<a target="_blank" href="https://colab.research.google.com/github/ChrisCurrin/goldilocks-GABA/blob/main/lrd_figures.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


**[Christopher B. Currin](https://chriscurrin.com)**<sup>1,2</sup>, Richard J. Burman<sup>1,3,4,5</sup>, Tommaso Fedele<sup>3</sup>, Georgia Ramantani<sup>3</sup>, Richard E. Rosch<sup>6,7</sup>, Henning Sprekeler<sup>8</sup>, and Joseph V. Raimondo<sup>1</sup>.

1. Division of Cell Biology, Department of Human Biology, Neuroscience Institute and Institute of Infectious Disease and Molecular Medicine, Faculty of Health Sciences, University of Cape Town, Cape Town, South Africa
2. Institute of Science and Technology Austria, Klosterneuburg, Austria
3. Department of Paediatric Neurology, University Children's Hospital Zurich and University of Zurich, Zurich, Switzerland
4. Department of Pharmacology, University of Oxford, United Kingdom
5. Oxford Epilepsy Research Group, Nuffield Department of Clinical Neurosciences, University of Oxford, Oxford, United Kingdom
6. Department of Clinical Neurophysiology, King’s College Hospital NHS Foundation Trust, London 
7. Wellcome Centre for Imaging Neuroscience, University College London, London, United Kingdom
8. Bernstein Center for Computational Neuroscience Berlin, Technische Universität Berlin, Marchstr 23, Berlin, Germany


## Abstract

> Seizures that continue for beyond five minutes are classified as status epilepticus (SE) and constitute a medical emergency. Benzodiazepines, the current first-line treatment, attempt to terminate SE by increasing the conductance of chloride-permeable type-A GABA receptors (GABA<sub>A</sub>Rs). Despite their widespread use, benzodiazepines are ineffective in over a third of cases. Previous research in animal models has demonstrated that changes in intraneuronal chloride homeostasis and GABAAR physiology may underlie the development of benzodiazepine resistance in SE. However, there remains a need to understand the effect of these changes at a network level to improve translation into the clinical domain. Therefore, informed by data from human EEG recordings of SE and experimental brain slice recordings, we used a large spiking neural network model that incorporates chloride dynamics to investigate and address the phenomenon of benzodiazepine resistance in SE. We found that the GABA<sub>A</sub>R reversal potential (EGABA) sets SE-like bursting and determines the response to GABA<sub>A</sub>R conductance modulation, with benzodiazepines being anti-seizure at low EGABA and ineffective or pro-seizure at high EGABA. The SE-like activity and EGABA depended on a non-linear relationship between the strength of Cl<sup>-</sup> extrusion and GABA<sub>A</sub>R conductance, but not on the initial EGABA of neurons. Independently controlling Cl<sup>-</sup> extrusion in the pyramidal and interneuronal cell populations revealed the critical role of pyramidal cell Cl<sup>-</sup> extrusion in determining the severity of SE activity and the response to simulated benzodiazepine application. Finally, we demonstrate the model’s utility for considering improved therapeutic approaches for terminating SE in the clinic.

## Dependencies

* Python (using conda recommended)

## Installation

* Clone this repo to your local machine
* `cd` into the repo directory
* Run `conda env create -f environment.yml` to create the environment
* Run `conda activate goldilocks-GABA` to activate the environment
* Run `python -m ipykernel install --user --name=goldilocks-GABA` to add the environment to Jupyter Notebook
* Run `jupyter notebook` to start the notebook server
* Navigate to the notebook in the browser window that opens

## Reproducing the results

Option 1: Run the cells in the notebook and click "Run" for the widgets

Option 2: Run each file in script
