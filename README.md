# Goldilocks GABA: Network models incorporating chloride dynamics predict optimal strategies for terminating status epilepticus

## Abstract

> A seizure that continues for more than 5 minutes constitutes a medical emergency and is classified as status epilepticus (SE). The current first-line treatments for SE target the conductance of the GABA type A receptor (GABAAR). Recent experiment data has demonstrated that agents which positively modulate GABAARs can have a paradoxical effect that depends on the duration of the seizure and the GABA reversal potential (EGABA). As further clinical and experimental research is conducted to understand this life-threatening brain state, I have sought to complement research efforts with an in silico network model that recapitulates experimental results and informs future therapeutic strategies. I demonstrate that 1) elevated EGABA is sufficient to have a balanced network enter an SE-like activity state. 2) The network bursting activity responds to benzodiazepine-like GABAAR modulation in agreement with experiments. 3) Compromised Cl- extrusion can lead a stable network into SE and maintain SE after Cl- accumulation, which can arise via a seizure. 4) Specifically targeting Cl- extrusion in pyramidal cells showcases their primary role in maintaining a network’s SE state. 5) The total GABAergic current leading up to SE bursts correlates with the severity of the state. This model forms a foundation for further theoretical research into this pathology. Together, my results back up experimental findings and suggest future nuanced approaches to treatment.

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
