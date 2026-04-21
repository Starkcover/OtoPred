# OtoTox Predictor — Streamlit Deployment

GNN + Random Forest ensemble for drug-induced ototoxicity prediction.
Accepts SMILES, ChEMBL ID, or InChIKey.

---

## Project structure

```
ototoxicity_streamlit/
├── app.py                        ← Streamlit app (single file)
├── requirements.txt              ← Python dependencies
├── packages.txt                  ← System packages (RDKit needs these)
├── .streamlit/
│   └── config.toml               ← Theme + server config
├── checkpoints/                  ← Model files (add manually — not in git)
│   ├── rf_baseline.pkl
│   ├── best_model_final.pt
│   ├── hetero_data_final.pt
│   ├── df_with_chembl.parquet
│   ├── protein_features_v2.npy
│   ├── protein_map_v2.parquet
│   ├── ensemble_config.pt
│   └── train_idx.npy
└── export_checkpoints.ipynb      ← Run on Kaggle to export model files
```

---

## Step 1 — Export model from Kaggle

1. Open `export_checkpoints.ipynb` in your Kaggle training session
2. Run all cells → downloads `ototox_checkpoints.zip`
3. Extract the zip → you get a `checkpoints/` folder
4. Copy that folder into `ototoxicity_streamlit/`

---

## Step 2 — Run locally

```bash
cd ototoxicity_streamlit
pip install -r requirements.txt
streamlit run app.py
# Opens at http://localhost:8501
```

---

## Step 3 — Deploy to Streamlit Community Cloud (free)

1. Push repo to GitHub:
   ```bash
   git init
   git add app.py requirements.txt packages.txt .streamlit/
   # DO NOT add checkpoints/ to git (files are too large)
   echo "checkpoints/" >> .gitignore
   git add .gitignore
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USER/ototox-predictor
   git push -u origin main
   ```

2. Go to https://share.streamlit.io → New app
3. Connect GitHub repo → select `app.py` → Deploy

4. **Add checkpoints via Streamlit secrets + cloud storage:**

   Since model files are large (>100MB), use one of:

   ### Option A — Hugging Face Hub (easiest for free)
   ```python
   # Upload checkpoints to HF:
   pip install huggingface_hub
   from huggingface_hub import HfApi
   api = HfApi()
   api.upload_folder(folder_path="checkpoints", repo_id="YOUR_HF_USERNAME/ototox-checkpoints", repo_type="dataset")
   ```
   Then in `app.py` add at startup:
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download(repo_id="YOUR_HF_USERNAME/ototox-checkpoints",
                     repo_type="dataset", local_dir="checkpoints")
   ```

   ### Option B — GitHub LFS
   ```bash
   git lfs install
   git lfs track "checkpoints/*.pt" "checkpoints/*.pkl" "checkpoints/*.parquet" "checkpoints/*.npy"
   git add .gitattributes checkpoints/
   git commit -m "Add model checkpoints via LFS"
   git push
   ```

   ### Option C — Just use RF-only mode
   RF predictions work without the GNN. Only add `rf_baseline.pkl` (usually < 50MB)
   and commit it directly to the repo.

---

## Environment variable

Set `CKPT_DIR` if your checkpoints are in a different location:
```toml
# .streamlit/secrets.toml
CKPT_DIR = "/path/to/checkpoints"
```

---

## Notes

- App falls back to **RF-only mode** automatically if GNN files are missing
- `packages.txt` installs system libraries needed by RDKit on Linux
- First load caches the model with `@st.cache_resource` — fast after that
