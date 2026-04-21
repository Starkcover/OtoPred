import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import joblib
import os
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Draw import rdMolDraw2D

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="OtoTox Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    max-width: 1100px !important;   /* controls overall width */
    margin: auto !important;        /* center horizontally */
    padding: 1.5rem 1rem !important; /* spacing inside */
}
.stApp { background: #f4f6f9; }

/* ── Header ── */
.app-header {
    background: white;
    border-bottom: 1px solid #e2e8f0;
    padding: 0 2rem;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 3px rgba(0,0,0,.07);
    position: sticky;
    top: 0;
    z-index: 999;
    margin-bottom: 1.5rem;
}
.header-left { display: flex; align-items: center; gap: 12px; }
.header-logo {
    width: 32px; height: 32px;
    background: #2563eb; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 14px; font-weight: 700;
    flex-shrink: 0;
}
.header-title { font-size: 15px; font-weight: 600; color: #1e293b; }
.header-sub { font-size: 11px; color: #64748b; }
.header-badge {
    font-size: 11px; padding: 3px 10px;
    background: #eff6ff; color: #2563eb;
    border-radius: 4px; font-weight: 500;
    border: 1px solid #bfdbfe;
}

/* ── Cards ── */
.card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,.07);
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.card-title {
    font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: .06em;
    color: #94a3b8; margin-bottom: 1rem;
    padding-bottom: .5rem;
    border-bottom: 1px solid #f1f5f9;
}

/* ── Verdict ── */
.verdict-tox {
    display: inline-flex; align-items: center; gap: 8px;
    background: #fee2e2; color: #dc2626;
    border: 1px solid #fca5a5; border-radius: 999px;
    padding: .4rem 1rem; font-size: 14px; font-weight: 600;
}
.verdict-safe {
    display: inline-flex; align-items: center; gap: 8px;
    background: #d1fae5; color: #059669;
    border: 1px solid #6ee7b7; border-radius: 999px;
    padding: .4rem 1rem; font-size: 14px; font-weight: 600;
}
.vdot-tox  { width:8px;height:8px;background:#dc2626;border-radius:50%;display:inline-block; }
.vdot-safe { width:8px;height:8px;background:#059669;border-radius:50%;display:inline-block; }

/* ── Prob bars ── */
.prob-wrap { margin-bottom: .8rem; }
.prob-meta { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:.3rem; }
.prob-name { font-size: 12px; color: #64748b; font-weight: 500; }
.prob-pct  { font-size: 13px; font-weight: 600; color: #1e293b; }
.track { background: #f1f5f9; border-radius: 999px; height: 6px; overflow: hidden; }
.fill-tox  { height:6px; border-radius:999px; background:linear-gradient(90deg,#ef4444,#f87171); }
.fill-safe { height:6px; border-radius:999px; background:linear-gradient(90deg,#10b981,#34d399); }
.fill-conf { height:6px; border-radius:999px; background:linear-gradient(90deg,#2563eb,#60a5fa); }
.fill-gnn  { height:6px; border-radius:999px; background:linear-gradient(90deg,#7c3aed,#a78bfa); }
.fill-rf   { height:6px; border-radius:999px; background:linear-gradient(90deg,#d97706,#fbbf24); }

/* ── Props grid ── */
.props-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:.5rem; margin-top:.5rem; }
.prop-card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:.55rem .75rem; }
.prop-lbl  { font-size:11px; color:#94a3b8; margin-bottom:.15rem; }
.prop-val  { font-size:14px; font-weight:600; color:#1e293b; }

/* ── Section title ── */
.sec-title {
    font-size:11px; font-weight:600; text-transform:uppercase;
    letter-spacing:.06em; color:#94a3b8;
    border-bottom:1px solid #e2e8f0;
    padding-bottom:.35rem; margin:1rem 0 .65rem;
}

/* ── Mol structure box ── */
.mol-box {
    background: white; border: 1px solid #e2e8f0;
    border-radius: 8px; overflow: hidden;
    text-align: center;
}
.mol-box img { max-width: 100%; height: auto; display: block; margin: 0 auto; }

/* ── SMILES code ── */
.smiles-code {
    font-family: monospace; font-size: 11px; color: #64748b;
    background: #f8fafc; border-radius: 6px;
    padding: .45rem .75rem; word-break: break-all;
    border: 1px solid #e2e8f0; margin-top: .5rem;
    line-height: 1.5;
}

/* ── Threshold note ── */
.thresh-note { font-size:11px; color:#94a3b8; margin-top:.2rem; line-height:1.5; }

/* ── Mode badge ── */
.mode-tag {
    font-size:11px; padding:.2rem .55rem;
    background:#f8fafc; color:#64748b;
    border:1px solid #e2e8f0; border-radius:4px;
    display:inline-block;
}

/* ── Example buttons ── */
.ex-row { display:flex; flex-wrap:wrap; gap:.4rem; }

/* ── Detail rows ── */
.detail-row {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: .35rem 0; border-bottom: 1px solid #f1f5f9;
    margin-bottom: .1rem;
}
.detail-row:last-child { border-bottom: none; }
.detail-lbl { font-size: 12px; color: #64748b; font-weight: 500; }
.detail-val { font-size: 13px; font-weight: 600; color: #1e293b; text-align: right; }
.detail-sub { font-size: 11px; color: #94a3b8; font-weight: 400; margin-left: .3rem; }

/* ── Percentage boxes ── */
.pct-box {
    border-radius: 10px; padding: .75rem 1rem;
    text-align: center; border: 1px solid transparent;
}
.pct-box-tox   { background: #fee2e2; border-color: #fca5a5; }
.pct-box-safe  { background: #d1fae5; border-color: #6ee7b7; }
.pct-box-neutral { background: #f1f5f9; border-color: #e2e8f0; }
.pct-box-label { font-size: 11px; color: #64748b; font-weight: 500; margin-bottom: .2rem; text-transform: uppercase; letter-spacing: .04em; }
.pct-box-val   { font-size: 22px; font-weight: 700; color: #1e293b; }

/* Override streamlit button */
div[data-testid="stButton"] button {
    border-radius: 999px !important;
    font-size: 12px !important;
    padding: .25rem .75rem !important;
    border: 1px solid #cbd5e1 !important;
    background: white !important;
    color: #64748b !important;
    font-weight: 500 !important;
}
div[data-testid="stButton"] button:hover {
    border-color: #2563eb !important;
    color: #2563eb !important;
    background: #eff6ff !important;
}

/* Main predict button */
div[data-testid="stButton"].predict-wrap button {
    background: #2563eb !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: .65rem !important;
    width: 100% !important;
}

/* Streamlit slider */
.stSlider [data-baseweb="slider"] { margin-top: -.5rem; }

/* Input labels */
.stTextInput label, .stTextArea label {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #64748b !important;
}
</style>
""", unsafe_allow_html=True)

# ── GNN architecture ──────────────────────────────────────────────────────────
CKPT_DIR = os.environ.get("CKPT_DIR", "checkpoints")

try:
    from torch_geometric.nn import SAGEConv, to_hetero
    from torch_geometric.data import HeteroData
    import torch.nn as nn

    class GNNv2(nn.Module):
        def __init__(self, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = SAGEConv((-1,-1), hidden_channels)
            self.bn1   = nn.BatchNorm1d(hidden_channels)
            self.conv2 = SAGEConv((-1,-1), hidden_channels)
            self.bn2   = nn.BatchNorm1d(hidden_channels)
            self.conv3 = SAGEConv((-1,-1), hidden_channels)
            self.bn3   = nn.BatchNorm1d(hidden_channels)
            self.lin   = nn.Linear(hidden_channels, out_channels)

        def forward(self, x, edge_index):
            x1 = self.bn1(self.conv1(x, edge_index)).relu()
            x1 = F.dropout(x1, p=0.3, training=self.training)
            x2 = self.bn2(self.conv2(x1, edge_index)).relu()
            x2 = F.dropout(x2, p=0.3, training=self.training)
            x3 = self.bn3(self.conv3(x2, edge_index)).relu()
            x3 = x3 + x1
            return self.lin(x3)

    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False

# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    result = {
        "rf": None, "gnn": None, "graph": None,
        "df": None, "train_idx": None,
        "best_w": 0.5, "threshold": 0.36,
        "mode": "unavailable", "train_fps": None, "valid_idx": None,
    }

    rf_path = f"{CKPT_DIR}/rf_baseline.pkl"
    if not os.path.exists(rf_path):
        return result
    result["rf"]   = joblib.load(rf_path)
    result["mode"] = "rf_only"

    cfg_path = f"{CKPT_DIR}/ensemble_config.pt"
    if os.path.exists(cfg_path):
        cfg = torch.load(cfg_path, map_location="cpu", weights_only=False)
        result["best_w"]    = float(cfg.get("best_w",    0.5))
        result["threshold"] = float(cfg.get("threshold", 0.36))

    gnn_path   = f"{CKPT_DIR}/best_model_final.pt"
    graph_path = f"{CKPT_DIR}/hetero_data_final.pt"
    df_path    = f"{CKPT_DIR}/df_with_chembl.parquet"

    if GNN_AVAILABLE and all(os.path.exists(p) for p in [gnn_path, graph_path, df_path]):
        try:
            graph = torch.load(graph_path, map_location="cpu", weights_only=False)
            gnn   = to_hetero(GNNv2(256, 2), graph.metadata())
            gnn.load_state_dict(torch.load(gnn_path, map_location="cpu", weights_only=False))
            gnn.eval()

            df = pd.read_parquet(df_path).reset_index(drop=True)

            split_path = f"{CKPT_DIR}/train_idx.npy"
            if os.path.exists(split_path):
                train_idx = np.load(split_path)
            else:
                train_idx = np.where(df["Data set"].str.strip().str.lower() == "training")[0]

            # Pre-build training fingerprints
            mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            fps, valid_idx = [], []
            for i in train_idx:
                smi = df.iloc[i]["canonical_smiles"]
                m   = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
                fp  = mfpgen.GetFingerprint(m) if m else None
                if fp:
                    fps.append(fp)
                    valid_idx.append(i)

            result.update({
                "gnn": gnn, "graph": graph, "df": df,
                "train_idx": train_idx,
                "train_fps": fps,
                "valid_idx": np.array(valid_idx),
                "mode": "ensemble",
            })
        except Exception as e:
            st.warning(f"GNN load failed: {e} — using RF only")

    return result


def mol_to_svg(mol, pred_class, size=(420, 280)):
    """Returns SVG string — works without Cairo on all platforms."""
    info = {}
    AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, bitInfo=info)
    hl_atoms, hl_colors = [], {}
    for bit in list(info.keys())[:5]:
        for atom_idx, _ in info[bit]:
            if atom_idx not in hl_atoms:
                hl_atoms.append(atom_idx)
                hl_colors[atom_idx] = (
                    (0.78, 0.27, 0.32) if pred_class == 1
                    else (0.22, 0.38, 0.70)
                )
    drawer = rdMolDraw2D.MolDraw2DSVG(*size)
    drawer.drawOptions().addStereoAnnotation = True
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer, mol,
        highlightAtoms=hl_atoms,
        highlightAtomColors=hl_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def resolve_smiles(smiles=None, chembl_id=None, inchikey=None):
    if smiles:
        return smiles.strip()
    if chembl_id or inchikey:
        try:
            from chembl_webresource_client.new_client import new_client
            m_api = new_client.molecule
            if chembl_id:
                res = m_api.filter(molecule_chembl_id=chembl_id.upper()).only(["molecule_structures"])
            else:
                res = m_api.filter(
                    molecule_structures__standard_inchi_key=inchikey
                ).only(["molecule_structures"])
            if res and res[0].get("molecule_structures"):
                return res[0]["molecule_structures"].get("canonical_smiles")
        except Exception as e:
            raise ValueError(f"ChEMBL lookup failed: {e}")
    return None


def predict(smiles, models, threshold):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    fp_arr = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048), dtype=np.float32)

    # RF
    rf_prob = float(models["rf"].predict_proba(fp_arr.reshape(1,-1))[0,1])

    # GNN
    gnn_prob  = None
    top_sim   = 0.0
    best_w    = models["best_w"]

    if models["gnn"] and models["train_fps"]:
        mfpgen  = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fp_rdk  = mfpgen.GetFingerprint(mol)
        sims    = np.array(DataStructs.BulkTanimotoSimilarity(fp_rdk, models["train_fps"]))
        topk    = np.argsort(sims)[::-1][:10]
        nbr_nodes = models["valid_idx"][topk]
        top_sim   = float(sims[topk[0]])

        graph = models["graph"]
        orig_x = graph["drug"].x.clone()
        graph["drug"].x[0] = torch.tensor(fp_arr)

        rel = ("drug","similar_to","drug")
        has_edges = rel in graph.edge_types
        orig_ei   = graph[rel].edge_index.clone() if has_edges else None

        src = [0]*len(nbr_nodes) + list(nbr_nodes)
        dst = list(nbr_nodes)   + [0]*len(nbr_nodes)
        new_ei = torch.tensor([src,dst], dtype=torch.long)
        if has_edges:
            graph[rel].edge_index = torch.cat([orig_ei, new_ei], dim=1)
        else:
            graph[rel].edge_index = new_ei

        with torch.no_grad():
            out      = models["gnn"](graph.x_dict, graph.edge_index_dict)
            gnn_prob = float(F.softmax(out["drug"][0], dim=0)[1])

        graph["drug"].x = orig_x
        if has_edges:
            graph[rel].edge_index = orig_ei
        elif hasattr(graph[rel], "edge_index"):
            del graph[rel].edge_index

    # Ensemble
    if gnn_prob is not None:
        ens_prob = best_w * gnn_prob + (1 - best_w) * rf_prob
    else:
        ens_prob = rf_prob
        best_w   = 0.0

    pred_class = 1 if ens_prob >= threshold else 0
    confidence = ens_prob if pred_class == 1 else (1 - ens_prob)

    # Mol props
    props = {
        "Mol. weight": round(Descriptors.MolWt(mol), 2),
        "LogP":        round(Descriptors.MolLogP(mol), 2),
        "TPSA":        f"{round(Descriptors.TPSA(mol), 1)} Å²",
        "HBA":         Chem.rdMolDescriptors.CalcNumHBA(mol),
        "HBD":         Chem.rdMolDescriptors.CalcNumHBD(mol),
        "Rot. bonds":  Chem.rdMolDescriptors.CalcNumRotatableBonds(mol),
    }

    mol_svg = mol_to_svg(mol, pred_class)

    return {
        "pred_class":  pred_class,
        "ens_prob":    round(ens_prob * 100, 1),
        "gnn_prob":    round((gnn_prob or 0) * 100, 1),
        "rf_prob":     round(rf_prob * 100, 1),
        "confidence":  round(confidence * 100, 1),
        "threshold":   round(threshold * 100, 1),
        "best_w":      round(best_w, 2),
        "rf_w":        round(1 - best_w, 2),
        "top_sim":     round(top_sim, 3),
        "smiles":      smiles,
        "mol_svg":     mol_svg,
        "props":       props,
        "mode":        "ensemble" if gnn_prob is not None else "rf_only",
    }


def prob_bar(name, pct, fill_cls, note=""):
    note_html = f'<p style="font-size:11px;color:#94a3b8;margin-top:.2rem">{note}</p>' if note else ""
    return f"""
    <div class="prob-wrap">
      <div class="prob-meta">
        <span class="prob-name">{name}</span>
        <span class="prob-pct">{pct}%</span>
      </div>
      <div class="track"><div class="{fill_cls}" style="width:{pct}%"></div></div>
      {note_html}
    </div>"""


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div class="app-header">
  <div class="header-left">
    <div class="header-logo">Ot</div>
    <div>
      <div class="header-title">OtoTox Predictor</div>
      <div class="header-sub">Drug-induced ototoxicity &nbsp;·&nbsp; GNN + Random Forest ensemble</div>
    </div>
  </div>
  <span class="header-badge">Research tool &nbsp;·&nbsp; Not for clinical use</span>
</div>
""", unsafe_allow_html=True)

# Load models
models = load_models()
model_ok = models["rf"] is not None

# ── Two-column layout ──────────────────────────────────────────────────────────
left, right = st.columns([4, 5], gap="medium")

with left:
    # Input card
    st.markdown('<div class="card"><div class="card-title">Drug Input</div>', unsafe_allow_html=True)

    input_tab = st.radio("Input method", ["SMILES string", "ChEMBL ID", "InChIKey"],
                         horizontal=True, label_visibility="collapsed")

    smiles_input   = ""
    chembl_input   = ""
    inchikey_input = ""

    if input_tab == "SMILES string":
        smiles_input = st.text_area("SMILES string", placeholder="e.g.  CC(=O)Oc1ccccc1C(=O)O",
                                    height=90, label_visibility="visible")
    elif input_tab == "ChEMBL ID":
        chembl_input = st.text_input("ChEMBL ID", placeholder="e.g.  CHEMBL25")
    else:
        inchikey_input = st.text_input("InChIKey", placeholder="e.g.  BSYNRYMUTXBXSQ-UHFFFAOYSA-N")

    st.markdown("---")

    # Threshold
    st.markdown("**Decision threshold**")
    threshold = st.slider(
        "threshold", min_value=0.20, max_value=0.80,
        value=models["threshold"], step=0.01,
        format="%.2f", label_visibility="collapsed"
    )
    col_s, col_v, col_sp = st.columns([1, 1, 1])
    with col_s: st.caption("← Sensitive")
    with col_v: st.markdown(f"<div style='text-align:center;font-weight:600;color:#2563eb;font-size:13px'>{threshold:.2f}</div>", unsafe_allow_html=True)
    with col_sp: st.caption("Specific →")
    st.markdown(f'<p class="thresh-note">Default {models["threshold"]:.2f} → 90% recall for ototoxic class.</p>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Predict button
    predict_clicked = st.button("🔬  Predict ototoxicity", use_container_width=True, type="primary")

    if not model_ok:
        st.error("⚠️ No model loaded. Add checkpoint files to the `checkpoints/` folder.")

    # Quick examples card
    st.markdown('<div class="card"><div class="card-title">Quick examples</div>', unsafe_allow_html=True)
    ex_cols = st.columns(5)
    examples = [("Cisplatin", "CHEMBL11359"), ("Gentamicin", "CHEMBL1464"),
                ("Aspirin", "CHEMBL25"), ("Furosemide", "CHEMBL1201067"),
                ("Ibuprofen", "CHEMBL112")]
    for col, (name, cid) in zip(ex_cols, examples):
        if col.button(name):
            st.session_state["example_chembl"] = cid
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Handle example click
    if "example_chembl" in st.session_state:
        chembl_input  = st.session_state.pop("example_chembl")
        input_tab     = "ChEMBL ID"
        predict_clicked = True


with right:
    # Run prediction
    if predict_clicked and model_ok:
        with st.spinner("Running prediction …"):
            try:
                resolved = resolve_smiles(
                    smiles=smiles_input or None,
                    chembl_id=chembl_input or None,
                    inchikey=inchikey_input or None,
                )
                if not resolved:
                    st.error("Could not resolve a SMILES. Check your input.")
                else:
                    r = predict(resolved, models, threshold)

                    tox       = r["pred_class"] == 1
                    pill_cls  = "verdict-tox" if tox else "verdict-safe"
                    dot_cls   = "vdot-tox"    if tox else "vdot-safe"
                    verdict   = "OTOTOXIC"    if tox else "NON-TOXIC"
                    fill_main = "fill-tox"    if tox else "fill-safe"
                    thresh_rel = "above" if tox else "below"

                    # ── 1. Verdict + summary card ─────────────────────────────
                    st.markdown(f"""
                    <div class="card">
                      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.1rem">
                        <div class="{pill_cls}">
                          <div class="{dot_cls}"></div>[{verdict}]
                        </div>
                        <span class="mode-tag">{r["mode"].replace("_"," ")}</span>
                      </div>

                      <div class="detail-row">
                        <span class="detail-lbl">Verdict</span>
                        <span class="detail-val" style="color:{'#dc2626' if tox else '#059669'};font-weight:600">[{verdict}]</span>
                      </div>
                      <div class="detail-row">
                        <span class="detail-lbl">Ensemble probability</span>
                        <span class="detail-val">{r["ens_prob"]}%
                          <span class="detail-sub">(threshold: {r["threshold"]}% · {thresh_rel})</span>
                        </span>
                      </div>
                      <div class="detail-row">
                        <span class="detail-lbl">Confidence</span>
                        <span class="detail-val">{r["confidence"]}%</span>
                      </div>
                      <div class="detail-row">
                        <span class="detail-lbl">Nearest training similarity</span>
                        <span class="detail-val">{r["top_sim"]}</span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── 2. Probability breakdown bars ─────────────────────────
                    st.markdown(f"""
                    <div class="card">
                      <div class="card-title">Probability breakdown</div>

                      <div style="display:grid;grid-template-columns:1fr 1fr;gap:.5rem;margin-bottom:1rem">
                        <div class="pct-box {'pct-box-tox' if tox else 'pct-box-safe'}">
                          <div class="pct-box-label">Ototoxic</div>
                          <div class="pct-box-val">{r["ens_prob"]}%</div>
                        </div>
                        <div class="pct-box pct-box-neutral">
                          <div class="pct-box-label">Non-toxic</div>
                          <div class="pct-box-val">{round(100 - r["ens_prob"], 1)}%</div>
                        </div>
                      </div>

                      {prob_bar("Ototoxic probability (ensemble)", r["ens_prob"], fill_main,
                                f"Threshold: {r['threshold']}% · {thresh_rel} threshold")}
                      {prob_bar("Confidence", r["confidence"], "fill-conf")}
                    </div>
                    """, unsafe_allow_html=True)

                    # ── 3. Model contribution breakdown ───────────────────────
                    if r["mode"] == "ensemble":
                        st.markdown(f"""
                        <div class="card">
                          <div class="card-title">Model contributions</div>

                          <div class="detail-row" style="margin-bottom:.6rem">
                            <span class="detail-lbl">GNN contribution</span>
                            <span class="detail-val">{r["gnn_prob"]}%
                              <span class="detail-sub">(weight: {r["best_w"]})</span>
                            </span>
                          </div>
                          {prob_bar("", r["gnn_prob"], "fill-gnn")}

                          <div class="detail-row" style="margin-top:.8rem;margin-bottom:.6rem">
                            <span class="detail-lbl">RF contribution</span>
                            <span class="detail-val">{r["rf_prob"]}%
                              <span class="detail-sub">(weight: {r["rf_w"]})</span>
                            </span>
                          </div>
                          {prob_bar("", r["rf_prob"], "fill-rf")}

                          <div class="detail-row" style="margin-top:.8rem">
                            <span class="detail-lbl">Ensemble (weighted)</span>
                            <span class="detail-val">{r["ens_prob"]}%
                              <span class="detail-sub">= {r["best_w"]} × GNN + {r["rf_w"]} × RF</span>
                            </span>
                          </div>
                          {prob_bar("", r["ens_prob"], fill_main)}
                        </div>
                        """, unsafe_allow_html=True)

                    # ── 4. Structure + properties ─────────────────────────────
                    col_mol, col_props = st.columns([3, 2])

                    with col_mol:
                        st.markdown('<div class="card"><div class="card-title">Structure</div>', unsafe_allow_html=True)
                        st.markdown(f'<div style="background:white;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden">{r["mol_svg"]}</div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="smiles-code">{r["smiles"]}</div>', unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    with col_props:
                        st.markdown('<div class="card"><div class="card-title">Molecular properties</div>', unsafe_allow_html=True)
                        for k, v in r["props"].items():
                            st.markdown(f"""
                            <div class="prop-card" style="margin-bottom:.4rem">
                              <div class="prop-lbl">{k}</div>
                              <div class="prop-val">{v}</div>
                            </div>""", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    elif not predict_clicked:
        # Placeholder
        st.markdown("""
        <div class="card" style="min-height:340px;display:flex;align-items:center;justify-content:center">
          <div style="text-align:center;color:#94a3b8;padding:2rem">
            <div style="font-size:3rem;margin-bottom:1rem">🔬</div>
            <div style="font-size:15px;font-weight:500;color:#64748b">Enter a drug to predict</div>
            <div style="font-size:13px;margin-top:.4rem">Results will appear here</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
