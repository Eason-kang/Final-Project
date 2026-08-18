# Dissertation Code

This folder contains the code and model artefacts needed for the dissertation analysis.

## Zhongbin Hu TPC Prediction Toolkit Files

Included with author permission and cited in the dissertation as Hu (2026):

- `code/OGT_predictor.py`
- `code/TPC_predictor.py`
- `code/FBA_anchor_point.py`

## Thesis Analysis Scripts

The scripts in `my_analysis_code/` are the key scripts used to connect the thesis data to the model pipeline:

- `select_soil_latitude_species.py`: selects soil-associated taxa, maps OTUs to species-level names and retrieves UniProt proteomes.
- `select_temp_gradient.py`: helper functions for BIOM parsing, taxonomy parsing, NCBI/UniProt lookup and FASTA download.
- `run_esm2_embeddings.py`: generates species-level ESM2 proteome embeddings.
- `run_tpc_batch.py`: runs OGT prediction and normalized TPC prediction using the model files in this folder.
- `run_fba_anchor_latitude.py`: runs the CarveMe + COBRApy FBA anchor step.
- `fit_fba_scaled_schoolfield.py`: fits Schoolfield-style traits to FBA-scaled predicted curves.
- `run_final_species_level_analysis.py`: generates final species-level statistical summaries and thesis figures.

## Citation
Hu, Z. (2026) *Microbial-Growth-TPC-Predictor: A toolkit for predicting microbial growth temperature-performance curves from genome- and proteome-level sequence information* [Computer software]. GitHub. Available at: https://github.com/Lakezh/Microbial-Growth-TPC-Predictor
