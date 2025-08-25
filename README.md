# **Uncertainty-LINE**: Length-Invariant Estimation of Uncertainty for Large Language Models.

This repository contains the code and processed data for reproducing the Uncertainty-LINE method.

---

## Repository Structure

```
├── processed_mans/                     # Processed experiment managers with all model outputs, UE scores, and quality metrics
├── results/                            # Collected experimental results in CSV format
├── plots/                              # Generated plots for analysis
│
├── 01_plotting_quality_ue_trends.ipynb # Notebook: analyze quality and uncertainty trends vs. generation length
├── 02_collect_experimental_results.ipynb # Notebook: aggregate and organize experimental results
├── 03_main_tables.ipynb                # Notebook: generate the main tables used in the paper
│
├── utils.py                            # Utility functions for data loading, detrending, and processing
├── enrich_metrics.py                   # Script to enrich generations with alternative quality metrics
│
├── README.md                           # Project documentation (this file)
└── requirements.txt                    # Python dependencies
```

---

## Usage

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt  
   ```

2. **Run lm-polygraph - collect data for training and test.**

   **Test split:**
   
   ```bash
   HYDRA_CONFIG=`pwd`/examples/configs/polygraph_eval_wmt14_csen.yaml \
     polygraph_eval \
     batch_size=1 \
     cache_path=/path/to/cache \
     model=gemma \
     subsample_eval_dataset=2000 \
     deberta_batch_size=1 \
     +deberta_device=cuda:0 \
     model.load_model_args.device_map=auto
   ```

   **Train split:**
   
   ```bash
   HYDRA_CONFIG=`pwd`/examples/configs/polygraph_eval_wmt14_csen.yaml \
      polygraph_eval \
      batch_size=1 \
      cache_path=/path/to/cache/train \
      model=gemma \
      subsample_eval_dataset=2000 \
      deberta_batch_size=1 \
      eval_split=train \
      +deberta_device=cuda:0 \
      model.load_model_args.device_map=auto
   ```




## Citation

If you use this repository, please cite:

```bibtex
@misc{vashurin2025uncertaintylinelengthinvariantestimationuncertainty,
      title={UNCERTAINTY-LINE: Length-Invariant Estimation of Uncertainty for Large Language Models}, 
      author={Roman Vashurin and Maiya Goloburda and Preslav Nakov and Maxim Panov},
      year={2025},
      eprint={2505.19060},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.19060}, 
}
```