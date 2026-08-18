# Dissertation Code

This folder contains the minimal code and model artefacts needed for the dissertation analysis.

## Zhongbin Hu TPC Prediction Toolkit Files

Included with author permission and cited in the dissertation as Hu (2026):

- `code/OGT_predictor.py`
- `code/TPC_predictor.py`
- `code/FBA_anchor_point.py`
- `results/core_model_checkpoint.pt`
- `results/core_model_scaler.pkl`
- `results/ogt_mlp/mlp.pkl`
- `results/ogt_mlp/scaler.pkl`
- `results/ogt_mlp/feature_cols.pkl`

The medium file `examples/example_medium_ecoli.json` is included because it is required for the FBA anchor step.

## Thesis Analysis Scripts

The scripts in `my_analysis_code/` are the key scripts used to connect the thesis data to the model pipeline:

- `select_soil_latitude_species.py`: selects soil-associated taxa, maps OTUs to species-level names and retrieves UniProt proteomes.
- `select_top150_species_temp_gradient.py`: helper functions for BIOM parsing, taxonomy parsing, NCBI/UniProt lookup and FASTA download.
- `run_manifest_esm2_embeddings.py`: generates species-level ESM2 proteome embeddings.
- `run_updated_tpc_batch.py`: runs OGT prediction and normalized TPC prediction using the model files in this folder.
- `run_fba_anchor_latitude.py`: runs the CarveMe + COBRApy FBA anchor step.
- `fit_fba_scaled_schoolfield.py`: fits Schoolfield-style traits to FBA-scaled predicted curves.
- `run_final_species_level_analysis.py`: generates final species-level statistical summaries and thesis figures.

## Citation

Hu, Z. (2026) *Microbial-Growth-TPC-Predictor: A toolkit for predicting microbial growth temperature-performance curves from genome- and proteome-level sequence information* [Computer software]. GitHub. Available at: https://github.com/Lakezh/Microbial-Growth-TPC-Predictor
