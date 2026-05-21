"""
DrugLens Screening Studio — Streamlit Web Application

Batch compound screening for early-stage kinase drug discovery.
Rebuilt around the workflow described in docs/SCREENING_STUDIO_BUILD_SPEC.md.
"""

import io
import html
import json
import datetime
import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

from src.features import featurize_pair, get_all_feature_names
from src.model import load_artifacts, predict_binding
from src.similarity import find_similar_drugs
from src.explainability import get_shap_explainer, explain_prediction, plot_shap_bar
from src.chemistry import (
    validate_smiles,
    canonicalize_smiles,
    smiles_to_image,
    compute_descriptors,
    count_lipinski_violations,
    parse_smiles_lines,
    parse_compound_csv,
    deduplicate_compounds,
)
from src.screening import (
    screen_compounds,
    assign_priority,
    summarize_invalid_compounds,
)

st.set_page_config(
    page_title="DrugLens Screening Studio",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Design System ────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=Source+Sans+3:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --cream: #faf8f5;
        --cream-dark: #f0ece6;
        --warm-white: #fdfcfa;
        --text-primary: #2d2a26;
        --text-secondary: #5c574f;
        --text-muted: #8a8078;
        --border: #e5e0d8;
        --border-light: #eee9e2;
        --accent: #c4705a;
        --accent-light: #d4845f;
        --accent-bg: #fdf0ec;
        --green: #5a9e6f;
        --green-bg: #eef6f0;
        --red: #c4705a;
        --red-bg: #fdf0ec;
        --blue: #5a7e9e;
        --blue-bg: #eef2f6;
        --yellow: #b8973e;
        --yellow-bg: #fdf8ec;
    }

    .stApp { background: var(--cream) !important; }
    .stApp > header { background: transparent !important; }

    .main .block-container {
        padding-top: 1.5rem !important;
        max-width: 1200px !important;
    }

    /* ── Sidebar ─────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--warm-white) !important;
        border-right: 1px solid var(--border) !important;
    }

    /* ── Metrics ─────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--cream) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
    }
    [data-testid="stMetricLabel"] p {
        color: var(--text-muted) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.65rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
    }
    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-family: 'Fraunces', serif !important;
        font-size: 1.3rem !important;
    }

    /* ── Typography ──────────────────────────────── */
    h1, h2, h3 {
        font-family: 'Fraunces', serif !important;
        color: var(--text-primary) !important;
    }
    p, li, label {
        font-family: 'Source Sans 3', sans-serif !important;
        color: var(--text-secondary) !important;
    }

    /* ── Inputs ──────────────────────────────────── */
    .stSelectbox > div > div,
    .stTextInput > div > div > input,
    .stTextArea textarea {
        background: var(--warm-white) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-family: 'Source Sans 3', sans-serif !important;
    }
    .stSelectbox label, .stTextInput label, .stTextArea label {
        color: var(--text-muted) !important;
        font-size: 0.85rem !important;
    }

    /* ── Button ──────────────────────────────────── */
    .stButton > button {
        background: var(--accent) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.7rem 2rem !important;
        letter-spacing: 0.01em !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: var(--accent-light) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(196, 112, 90, 0.25) !important;
    }

    /* ── Download button ─────────────────────────── */
    .stDownloadButton > button {
        background: var(--warm-white) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-family: 'Source Sans 3', sans-serif !important;
        font-weight: 500 !important;
        font-size: 0.85rem !important;
    }
    .stDownloadButton > button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }

    /* ── Expander ────────────────────────────────── */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary {
        background: var(--cream) !important;
        border-radius: 8px !important;
        font-family: 'Source Sans 3', sans-serif !important;
        color: var(--text-secondary) !important;
        padding: 0.6rem 1rem !important;
    }
    .streamlit-expanderContent,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background: var(--warm-white) !important;
        border: 1px solid var(--border-light) !important;
    }

    /* ── Code & Dividers ─────────────────────────── */
    code, .stCode { font-family: 'IBM Plex Mono', monospace !important; font-size: 0.8rem !important; }
    hr { border-color: var(--border-light) !important; }
    .stAlert { background: var(--cream) !important; border: 1px solid var(--border) !important; border-radius: 8px !important; }

    /* ── Custom components ────────────────────────── */
    .dl-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-bottom: 0.25rem;
    }
    .dl-title {
        font-family: 'Fraunces', serif;
        font-size: 1.35rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 0.75rem;
    }
    .dl-card {
        background: var(--warm-white);
        border: 1px solid var(--border-light);
        border-radius: 10px;
        padding: 1.25rem;
    }
    .dl-mol-wrap {
        background: white;
        border: 1px solid var(--border-light);
        border-radius: 8px;
        padding: 0.25rem;
        display: flex;
        justify-content: center;
    }

    /* ── Priority badges ─────────────────────────── */
    .dl-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.02em;
    }
    .dl-badge-high { background: var(--green-bg); color: var(--green); }
    .dl-badge-review { background: var(--yellow-bg); color: var(--yellow); }
    .dl-badge-low { background: var(--red-bg); color: var(--red); }

    /* ── Stat boxes ──────────────────────────────── */
    .dl-stat {
        background: var(--warm-white);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        padding: 0.9rem;
        text-align: center;
        margin-bottom: 0.6rem;
    }
    .dl-stat-val {
        font-family: 'Fraunces', serif;
        font-size: 1.15rem;
        color: var(--text-primary);
        font-weight: 600;
    }
    .dl-stat-lbl {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.6rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.15rem;
    }

    /* ── Feature list ────────────────────────────── */
    .dl-feat-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.45rem 0;
        border-bottom: 1px solid var(--border-light);
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.85rem;
    }
    .dl-feat-row:last-child { border-bottom: none; }
    .dl-feat-name { color: var(--text-secondary); }
    .dl-feat-pos { color: var(--green); font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; font-weight: 500; }
    .dl-feat-neg { color: var(--red); font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; font-weight: 500; }

    /* ── Similar drugs ───────────────────────────── */
    .dl-sim {
        background: var(--warm-white);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.5rem;
    }
    .dl-sim-score { font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; color: var(--text-primary); font-weight: 500; }
    .dl-sim-status { font-family: 'Source Sans 3', sans-serif; font-size: 0.8rem; }
    .dl-sim-smiles { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; color: var(--text-muted); margin-top: 0.2rem; word-break: break-all; }

    /* ── Sidebar ──────────────────────────────────── */
    .dl-sb-brand {
        padding: 0.25rem 0 1.25rem 0;
        border-bottom: 1px solid var(--border-light);
        margin-bottom: 1rem;
    }
    .dl-sb-name { font-family: 'Fraunces', serif; font-size: 1.5rem; font-weight: 700; color: var(--text-primary); }
    .dl-sb-tag { font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.12em; }
    .dl-sb-data {
        background: var(--cream);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        padding: 0.9rem;
    }
    .dl-sb-data-row {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0;
        font-size: 0.8rem;
        border-bottom: 1px solid var(--border-light);
    }
    .dl-sb-data-row:last-child { border-bottom: none; }
    .dl-sb-data-key { font-family: 'Source Sans 3', sans-serif; color: var(--text-muted); }
    .dl-sb-data-val { font-family: 'IBM Plex Mono', monospace; color: var(--text-primary); font-weight: 500; }

    /* ── Author ───────────────────────────────────── */
    .dl-author {
        background: var(--cream);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1.5rem;
    }
    .dl-author-name { font-family: 'Fraunces', serif; font-size: 0.9rem; font-weight: 600; color: var(--text-primary); }
    .dl-author-info { font-family: 'Source Sans 3', sans-serif; font-size: 0.75rem; color: var(--text-muted); line-height: 1.6; }
    .dl-author a { color: var(--accent); text-decoration: none; font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; }
    .dl-author a:hover { text-decoration: underline; }

    /* ── Result card ──────────────────────────────── */
    .dl-result { border-radius: 10px; padding: 1.75rem; text-align: center; position: relative; }
    .dl-result-binding { background: var(--green-bg); border: 1px solid rgba(90, 158, 111, 0.25); }
    .dl-result-nonbinding { background: var(--red-bg); border: 1px solid rgba(196, 112, 90, 0.25); }
    .dl-result-eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.2em; }
    .dl-result-eyebrow-bind { color: var(--green); }
    .dl-result-eyebrow-nobind { color: var(--red); }
    .dl-result-verdict { font-family: 'Fraunces', serif; font-size: 1.5rem; font-weight: 600; margin: 0.4rem 0 0.2rem 0; }
    .dl-result-verdict-bind { color: var(--green); }
    .dl-result-verdict-nobind { color: var(--red); }
    .dl-result-num { font-family: 'Fraunces', serif; font-size: 2.8rem; font-weight: 700; line-height: 1; }
    .dl-result-num-bind { color: var(--green); }
    .dl-result-num-nobind { color: var(--red); }
    .dl-result-sub { font-family: 'Source Sans 3', sans-serif; font-size: 0.8rem; color: var(--text-muted); margin-top: 0.3rem; }

    /* ── Hide nonessential Streamlit chrome ───────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: visible;}
</style>
""", unsafe_allow_html=True)


# ─── Data ──────────────────────────────────────────────────────────────────

import json as _json

def _load_davis_proteins() -> dict:
    _path = Path("data/proteins")
    if _path.exists():
        with open(_path) as f:
            return _json.load(f)
    return {}

_DAVIS_PROTEINS = _load_davis_proteins()

_KINASE_TARGET_META = {
    "EGFR — non-small cell lung cancer": ("EGFR", "Key target in lung cancer. Known drugs: Erlotinib, Gefitinib."),
    "CDK2 — cell cycle regulation": ("CDK2", "Cell cycle regulator. Validated cancer drug target."),
    "BRAF V600E — melanoma": ("BRAF(V600E)", "Melanoma oncogene. Drugs: Vemurafenib, Dabrafenib."),
    "ABL1 — chronic myeloid leukemia": ("ABL1", "CML oncogene. Drugs: Imatinib, Dasatinib, Nilotinib."),
    "ALK — non-small cell lung cancer": ("ALK", "Lung cancer fusion target. Drugs: Crizotinib, Alectinib."),
    "AURKA — mitotic kinase": ("AURKA", "Mitotic regulator, cancer drug target. Drug: Alisertib."),
    "BTK — B-cell malignancies": ("BTK", "B-cell signaling kinase. Drugs: Ibrutinib, Acalabrutinib."),
    "ERBB2/HER2 — breast cancer": ("ERBB2", "Breast cancer oncogene. Drugs: Lapatinib, Neratinib."),
    "FGFR1 — bladder & cholangiocarcinoma": ("FGFR1", "Fibroblast growth factor receptor. Drug: Erdafitinib."),
    "FLT3 — acute myeloid leukemia": ("FLT3", "AML target. Drugs: Midostaurin, Gilteritinib."),
    "KIT — gastrointestinal stromal tumor": ("KIT", "GIST oncogene. Drugs: Imatinib, Sunitinib."),
    "JAK2 — myeloproliferative neoplasms": ("JAK2(JH1domain-catalytic)", "JAK-STAT signaling. Drug: Ruxolitinib."),
    "MEK1 — melanoma & NSCLC": ("MEK1", "MAPK pathway. Drugs: Trametinib, Cobimetinib."),
    "RET — thyroid & lung cancer": ("RET", "RET-fusion cancers. Drugs: Selpercatinib, Pralsetinib."),
    "SRC — solid tumors": ("SRC", "Proto-oncogene kinase. Drug: Dasatinib."),
    "VEGFR2 — angiogenesis": ("VEGFR2", "Anti-angiogenic target. Drugs: Sorafenib, Sunitinib, Axitinib."),
}

FEATURED_KINASE_TARGETS = {}
for _display, (_davis_key, _desc) in _KINASE_TARGET_META.items():
    if _davis_key in _DAVIS_PROTEINS:
        FEATURED_KINASE_TARGETS[_display] = {
            "sequence": _DAVIS_PROTEINS[_davis_key],
            "description": _desc,
            "domain": "kinase",
            "in_training": True,
        }

def _build_all_davis_targets() -> dict:
    wild_type = {}
    mutants = {}
    for key, seq in _DAVIS_PROTEINS.items():
        if key in {v[0] for v in _KINASE_TARGET_META.values()}:
            continue
        entry = {
            "sequence": seq,
            "description": "Davis dataset kinase target.",
            "domain": "kinase",
            "in_training": True,
        }
        if "(" in key:
            mutants[key] = entry
        else:
            wild_type[key] = entry
    wt_sorted = dict(sorted(wild_type.items()))
    mut_sorted = dict(sorted(mutants.items()))
    return {**FEATURED_KINASE_TARGETS, **wt_sorted, **mut_sorted}

ALL_DAVIS_TARGETS = _build_all_davis_targets()

OUT_OF_DOMAIN_TARGETS = {
    "ACE2 — SARS-CoV-2 entry receptor": {
        "sequence": "MSSSSWLLLSLVAVTAAQSTIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQQNGSSVLSEDKSKRLNTILNTMSTIYSTGKVCNPDNPQECLLLEPGLNEIMANSLDYNERLWAWESWRSEVGKQLRPLYEEYVVLKNEMARANHYEDYGDYWRGDYEVNGVDGYDYSRGQLIEDVEHTFEEIKPLYEHLHAYVRAKLMNAYPSYISPIGCLPAHLLGDMWGRFWTNLYSLTVPFGQKPNIDVTDAMVDQAWDAQRIFKEAEKFFVSVGLPNMTQGFWENSMLTDPGNVQKAVCHPTAWDLGKGDFRILMCTKVTMDDFLTAHHEMGHIQYDMAYAAQPFLLRNGANEGFHEAVGEIMSLSAATPKHLKSIGLLSPDFQEDNETEINFLLKQALTIVGTLPFTYMLEKWRWMVFKGEIPKDQWMKKWWEMKREIVGVVEPVPHDETYCDPASLFHVSNDYSFIRYYTRTLYQFQFQEALCQAAKHEGPLHKCDISNSTEAGQKLFNMLRLGKSEPWTLALENVVGAKNMNVRPLLNYFEPLFTWLKDQNKNSFVGWSTDWSPYAD",
        "description": "COVID-19 viral entry point. Target for antiviral therapeutics.",
        "domain": "out-of-domain",
        "in_training": False,
    },
    "COX-2 — inflammation & pain": {
        "sequence": "MLARALLLCAVLALSHTANPCCSHPCQNRGVCMSVGFDQYKCDCTRTGFYGENCSTPEFLTRIKLFLKPTPNTVHYILTHFKGFWNVVNNIPFLRNAIMSYVLTSRSHLIDSPPTYNADYGYKSWEAFSNLSYYTRALPPVPDDCPTPLGVKGKKQLPDSNEIVEKLLLRRKFIPDPQGSNMMFAFFAQHFTHQFFKTDHKRGPAFTNGLGHGVDLNHIYGETLARQRKLRLFKDGKMKYQIIDGEMYPPTVKDTQAEMIYPPQVPEHLRFAVGQEVFGLVPGLMMYATIWLREHNRVCDVLKQEHPEWGDEQLFQTTRLILIGETIKIVIEDYVQHLSGYHFKLKFDPELLFNKQFQYQNRIAAEFNTLYHWHPLLPDTFQIHDQKYNYQQFIYNNSILLEHGITQFVESFTRQIAGRVAGGRNVPPAVQKVSQASIDQSRQMKYQSFNEYRKRFMLKPYESFEELTGEKEMSAELEALYGDIDAVELYPALLVEKPRPDAIFGETMVEVGAPFSLKGLMGNVICSPAYWKPSTFGGEVGFQIINTASIQSLICNNVKGCPFTSFSVPDPELIKTVTINASSSRSGLDDINPTVLLKERSTEL",
        "description": "Anti-inflammatory target. Drugs: Celecoxib, Ibuprofen.",
        "domain": "out-of-domain",
        "in_training": False,
    },
    "DPP-4 — type 2 diabetes": {
        "sequence": "MKTPWKVLLGLLGAAALVTIITVPVVLLNKGTDDATADSRKTYTLTDYLKNTYRLKLYSLRWISDHEYLYKQENNILVFNAEYGNSSVFLENSTFDEFGHSINDYSISPDGQFILLEYNYVKQWRHSYTASYDIYDLNKRQLITEERIPNNTQWVTWSPVGHKLAYVWNNDIYVKIEPNLPSYRITWTGKEDIIYNGITDWVYEEEVFSAYSALWWSPNGTFLAYAQFNDTEVPLIEYSFYSDESLQYPKTVRVPYPKAGAVNPTVKFFVVNTDSLSSVTNATSIQITAPASMLIGDHYLCDVTWATQERISLQWLRRIQNYSVMDICDYDESSGRWNCLVARQHIEMSTTGWVGRFRPSEPHFTLDGNSFYKIISNEEGYRHICYFQIDKKDCTFITKGTWEVIGIEALTSDYLYYISNEYKGMPGGRNLYKIQLSDYTKVTCLSCELNPERCQYYSVSFSKEAKYYQLRCSGPGLPLYTLHSSVNDKGLRVLEDNSALDKMLQNVQMPSKKLDFIILNETKFWYQMILPPHFDKSKKYPLLLDVYAGPCSQKADTVFRLNWATYLASTENIIVASFDGRGSGYQGDKIMHAINRRLGTFEVEDQIEAARQFSKMGFVDNKRIAIWGWSYGGYVTSMVLGSGSGVFKCGIAVAPVSRWEYYDSVYTERYMGLPTPEDNLDHYRNSTVMSRAENFKQVEYLLIHGTADDNVHFQQSAQISKALVDVGVDFQAMWYTDEDHGIASSTAHQHIYTHMSHFIKQCFSLP",
        "description": "Type 2 diabetes target. Drugs: Sitagliptin (Januvia).",
        "domain": "out-of-domain",
        "in_training": False,
    },
}

EXAMPLE_TARGETS = {**ALL_DAVIS_TARGETS, **OUT_OF_DOMAIN_TARGETS}

EXAMPLE_COMPOUNDS = {
    "Staurosporine": "O=C1NC2=CC=CC=C2C1=CC3=CN=C4C=CC=CC34",
    "Erlotinib": "COCCOC1=CC2=C(C=C1OCCOC)C(=NC=N2)NC3=CC=CC(=C3)C#C",
    "Imatinib": "CN1CCN(CC1)CC2=CC=C(C=C2)C(=O)NC3=CC(=C(C=C3)NC4=NC=CC(=N4)C5=CN=CC=C5)C",
    "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Metformin": "CN(C)C(=N)NC(=N)N",
    "Gefitinib": "COC1=CC2=C(C=C1OC)C(=NC=N2)NC3=CC(=C(C=C3)F)Cl",
    "Sorafenib": "CNC(=O)C1=CC(=C(C=C1)OC2=CC=C(C=C2)NC(=O)NC3=CC(=C(C=C3)Cl)C(F)(F)F)F",
    "Vemurafenib": "CCCS(=O)(=O)NC1=CC(=C(C=C1F)C(=O)C2=CNC3=NC=C(C=C23)C4=CC=C(C=C4)Cl)F",
    "Lapatinib": "CS(=O)(=O)CCNCC1=CC=C(O1)C2=CC3=C(C=C2)NC(=N3)/C=C/C4=CC=C(C=C4)Cl",
    "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
}


@st.cache_resource
def load_model():
    """Load model artifacts once."""
    return load_artifacts("artifacts")


@st.cache_data
def load_davis_compounds() -> dict[str, str]:
    """Load the local Davis ligand library if available."""
    path = Path("data/ligands")
    if not path.exists():
        return {}

    with open(path, "r") as f:
        raw = json.load(f)

    compounds = {}
    for name, smiles in raw.items():
        label = f"Davis {name}"
        compounds[label] = smiles
    return compounds


def clean_feature_name(name: str) -> str:
    """Human-readable feature name."""
    if name.startswith("MorganBit_"):
        return f"Substructure #{name.split('_')[1]}"
    return name.replace("_", " ")


def priority_badge_html(priority: str) -> str:
    """Return HTML badge for a priority bucket."""
    if priority == "High priority":
        return '<span class="dl-badge dl-badge-high">High priority</span>'
    elif priority == "Review":
        return '<span class="dl-badge dl-badge-review">Review</span>'
    else:
        return '<span class="dl-badge dl-badge-low">Low priority</span>'


def generate_html_report(
    filtered_df: pd.DataFrame,
    results_df: pd.DataFrame,
    target_name: str,
    target_info: dict,
    metrics: dict,
    screening_time: str,
    is_kinase: bool,
) -> str:
    esc = html.escape

    top5 = filtered_df.head(5)
    str_cols = {"Name", "SMILES", "Priority", "Trust", "Trust Reason", "Nearest Known Compound", "Nearest Known Label"}
    display_cols = [
        "Name", "SMILES", "Binding Prob", "Priority", "Trust", "Trust Reason",
        "Lipinski Violations", "MW", "LogP", "TPSA",
        "Nearest Known Compound", "Nearest Known Label", "Similar Known Score",
    ]
    available = [c for c in display_cols if c in filtered_df.columns]

    def _row_html(row):
        cells = []
        for c in available:
            val = row.get(c, "")
            cells.append(f"<td>{esc(str(val)) if c in str_cols else val}</td>")
        return f"<tr>{''.join(cells)}</tr>\n"

    top5_rows = "".join(_row_html(row) for _, row in top5.iterrows())
    all_rows = "".join(_row_html(row) for _, row in filtered_df.iterrows())

    header_cells = "".join(f"<th>{esc(c)}</th>" for c in available)

    n_high = (filtered_df["Priority"] == "High priority").sum() if not filtered_df.empty else 0
    median_prob = filtered_df["Binding Prob"].median() if not filtered_df.empty else 0.0
    n_lipinski_clean = (filtered_df["Lipinski Violations"].fillna(99).astype(int) == 0).sum() if not filtered_df.empty else 0

    safe_target = esc(target_name)
    ood_warning = ""
    if not is_kinase:
        short = esc(target_name.split(" — ")[0])
        ood_warning = f"""
        <div class="warning">
            <strong>{short}</strong> is not a kinase. This model was trained on the Davis
            kinase dataset and predictions for non-kinase targets may be unreliable.
        </div>"""

    ds = metrics.get("dataset_stats", {})

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DrugLens Screening Report — {safe_target}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #faf8f5; color: #2d2a26; padding: 2rem; max-width: 1000px; margin: 0 auto; line-height: 1.5; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.2rem; color: #5c574f; margin: 2rem 0 0.75rem 0; border-bottom: 1px solid #e5e0d8; padding-bottom: 0.4rem; }}
  .subtitle {{ color: #8a8078; font-size: 0.9rem; margin-bottom: 1.5rem; }}
  .meta {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; margin-bottom: 1.5rem; }}
  .meta-card {{ background: #fdfcfa; border: 1px solid #eee9e2; border-radius: 8px; padding: 0.75rem 1rem; }}
  .meta-label {{ font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em; color: #8a8078; }}
  .meta-value {{ font-size: 1.1rem; font-weight: 600; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; margin-top: 0.5rem; }}
  th {{ background: #f0ece6; text-align: left; padding: 0.5rem 0.75rem; font-weight: 600; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #5c574f; }}
  td {{ padding: 0.5rem 0.75rem; border-bottom: 1px solid #eee9e2; }}
  tr:hover {{ background: #fdfcfa; }}
  .warning {{ background: #fdf0ec; border: 1px solid #d4845f; border-radius: 8px; padding: 0.75rem 1rem; color: #c4705a; margin-bottom: 1rem; font-size: 0.85rem; }}
  .caveat {{ font-size: 0.75rem; color: #8a8078; text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #eee9e2; }}
  .badge {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 10px; font-size: 0.7rem; font-weight: 500; }}
  .badge-high {{ background: #eef6f0; color: #5a9e6f; }}
  .badge-review {{ background: #fdf8ec; color: #b8973e; }}
  .badge-low {{ background: #fdf0ec; color: #c4705a; }}
  @media print {{ body {{ padding: 0.5rem; }} }}
</style>
</head>
<body>
<h1>DrugLens Screening Report</h1>
<p class="subtitle">Generated {screening_time} &middot; Computational prioritization only</p>

{ood_warning}

<h2>Target</h2>
<div class="meta">
  <div class="meta-card"><div class="meta-label">Target</div><div class="meta-value">{safe_target}</div></div>
  <div class="meta-card"><div class="meta-label">Residues</div><div class="meta-value">{len(target_info.get('sequence', '')):,}</div></div>
  <div class="meta-card"><div class="meta-label">Domain</div><div class="meta-value">{'Kinase (in-domain)' if is_kinase else 'Out-of-domain'}</div></div>
</div>

<h2>Model</h2>
<div class="meta">
  <div class="meta-card"><div class="meta-label">AUROC</div><div class="meta-value">{metrics['auroc']:.4f}</div></div>
  <div class="meta-card"><div class="meta-label">F1</div><div class="meta-value">{metrics['f1']:.4f}</div></div>
  <div class="meta-card"><div class="meta-label">Precision</div><div class="meta-value">{metrics['precision']:.4f}</div></div>
  <div class="meta-card"><div class="meta-label">Recall</div><div class="meta-value">{metrics['recall']:.4f}</div></div>
  <div class="meta-card"><div class="meta-label">Dataset</div><div class="meta-value">Davis (TDC)</div></div>
  <div class="meta-card"><div class="meta-label">Training pairs</div><div class="meta-value">{ds.get('total_pairs', 0):,}</div></div>
</div>

<h2>Screening Summary</h2>
<div class="meta">
  <div class="meta-card"><div class="meta-label">Total screened</div><div class="meta-value">{len(results_df)}</div></div>
  <div class="meta-card"><div class="meta-label">After filters</div><div class="meta-value">{len(filtered_df)}</div></div>
  <div class="meta-card"><div class="meta-label">High priority</div><div class="meta-value">{n_high}</div></div>
  <div class="meta-card"><div class="meta-label">Median prob</div><div class="meta-value">{median_prob:.3f}</div></div>
  <div class="meta-card"><div class="meta-label">Lipinski clean</div><div class="meta-value">{n_lipinski_clean}</div></div>
</div>

<h2>Top 5 Compounds</h2>
<table>
  <thead><tr>{header_cells}</tr></thead>
  <tbody>{top5_rows}</tbody>
</table>

<h2>All Screened Compounds ({len(filtered_df)})</h2>
<table>
  <thead><tr>{header_cells}</tr></thead>
  <tbody>{all_rows}</tbody>
</table>

<h2>Limitations</h2>
<ul style="font-size: 0.85rem; color: #5c574f; padding-left: 1.5rem; margin-top: 0.5rem;">
  <li>Model trained on Davis kinase dataset (~{ds.get('unique_drugs', 0)} drugs, ~{ds.get('unique_targets', 0)} targets). Predictions outside this chemical/target space are extrapolations.</li>
  <li>Binary classification (binds/doesn't bind), not binding affinity regression.</li>
  <li>Molecular fingerprint + protein descriptor features — no 3D structural information.</li>
  <li>Predictions are computational prioritization for experimental follow-up, not clinical evidence.</li>
</ul>

<p class="caveat">
  DrugLens predictions are computational prioritization based on machine learning.
  They do not constitute experimental validation, medical advice, or clinical evidence.
  All candidate compounds require laboratory verification.
</p>
</body>
</html>"""


# ─── Main Application ─────────────────────────────────────────────────────

def main():
    if not Path("artifacts/model.joblib").exists():
        st.error("Model artifacts not found. Run `python train.py` first.")
        return

    model, metrics, ref_db = load_model()
    feature_names = get_all_feature_names()

    # ── Sidebar ────────────────────────────────────
    with st.sidebar:
        st.markdown("""
            <div class="dl-sb-brand">
                <div class="dl-sb-name">DrugLens</div>
                <div class="dl-sb-tag">Screening Studio</div>
            </div>
        """, unsafe_allow_html=True)

        # ── Demo presets ──
        _DEMO_PRESETS = {
            "Custom": None,
            "Quick demo — 10 curated vs CDK2": {
                "target_group": "Featured kinases",
                "target": "CDK2 — cell cycle regulation",
                "example_mode": "Curated examples",
                "count": 10,
            },
            "In-domain batch — Davis ligands vs EGFR": {
                "target_group": "Featured kinases",
                "target": "EGFR — non-small cell lung cancer",
                "example_mode": "Davis training ligands",
                "count": 68,
            },
            "Out-of-domain — 10 curated vs ACE2": {
                "target_group": "Out-of-domain examples",
                "target": "ACE2 — SARS-CoV-2 entry receptor",
                "example_mode": "Curated examples",
                "count": 10,
            },
        }

        def _apply_preset():
            p = _DEMO_PRESETS.get(st.session_state.get("_preset_select"))
            if p:
                st.session_state["_target_group"] = p["target_group"]
                st.session_state["_target_select"] = p["target"]
                st.session_state["_compound_source"] = "Example library"
                st.session_state["_example_mode"] = p["example_mode"]
                examples = EXAMPLE_COMPOUNDS if p["example_mode"] == "Curated examples" else load_davis_compounds()
                st.session_state["_selected_examples"] = list(examples.keys())[:p["count"]]

        st.selectbox(
            "Demo preset",
            list(_DEMO_PRESETS.keys()),
            key="_preset_select",
            on_change=_apply_preset,
        )

        st.markdown("---")

        # ── Target selection ──
        st.markdown("### Target protein")
        target_group = st.radio(
            "Target group",
            ["Featured kinases", "All Davis targets", "Out-of-domain examples"],
            horizontal=True,
            key="_target_group",
        )
        _is_kinase = target_group != "Out-of-domain examples"
        if target_group == "Featured kinases":
            target_pool = FEATURED_KINASE_TARGETS
        elif target_group == "All Davis targets":
            target_pool = ALL_DAVIS_TARGETS
        else:
            target_pool = OUT_OF_DOMAIN_TARGETS

        target_choice = st.selectbox(
            "Select target",
            list(target_pool.keys()),
            label_visibility="collapsed",
            key="_target_select",
        )
        if target_group == "All Davis targets":
            st.caption(f"{len(target_pool)} Davis targets · type to search")

        target_info = target_pool[target_choice]
        st.caption(target_info["description"])
        _in_training = target_info.get("in_training", False)
        st.caption(f"{len(target_info['sequence']):,} residues · In training set: {'Yes' if _in_training else 'No'}")

        if not _is_kinase:
            _target_short = target_choice.split(" — ")[0]
            st.warning(
                f"**{_target_short}** is not a kinase. This model was trained "
                "on the Davis kinase dataset and predictions for non-kinase "
                "targets may be unreliable.",
            )

        # Clear stale results when the target changes
        if ("results_df" in st.session_state
                and st.session_state.get("target_name") != target_choice):
            for key in ["results_df", "target_name", "target_info", "screening_time"]:
                st.session_state.pop(key, None)

        st.markdown("---")

        # ── Filters ──
        st.markdown("### Filters")
        min_binding = st.slider(
            "Min binding probability",
            min_value=0.0, max_value=1.0, value=0.0, step=0.05,
            help="Only show compounds above this binding probability.",
        )
        max_lipinski = st.slider(
            "Max Lipinski violations",
            min_value=0, max_value=4, value=4, step=1,
            help="Filter out compounds exceeding this many Lipinski rule-of-five violations.",
        )
        mw_range = st.slider(
            "Molecular weight range",
            min_value=0, max_value=1500, value=(0, 1500), step=50,
        )
        logp_range = st.slider(
            "LogP range",
            min_value=-5.0, max_value=15.0, value=(-5.0, 15.0), step=0.5,
        )

        st.markdown("---")

        # ── Model metrics ──
        st.markdown("### Model performance")
        c1, c2 = st.columns(2)
        c1.metric("AUROC", f"{metrics['auroc']:.3f}")
        c2.metric("F1", f"{metrics['f1']:.3f}")
        c3, c4 = st.columns(2)
        c3.metric("Precision", f"{metrics['precision']:.3f}")
        c4.metric("Recall", f"{metrics['recall']:.3f}")

        ds = metrics.get("dataset_stats", {})
        st.markdown("### Dataset")
        st.markdown(f"""
            <div class="dl-sb-data">
                <div class="dl-sb-data-row"><span class="dl-sb-data-key">Source</span><span class="dl-sb-data-val">Davis (TDC)</span></div>
                <div class="dl-sb-data-row"><span class="dl-sb-data-key">Pairs</span><span class="dl-sb-data-val">{ds.get('total_pairs', 0):,}</span></div>
                <div class="dl-sb-data-row"><span class="dl-sb-data-key">Drugs</span><span class="dl-sb-data-val">{ds.get('unique_drugs', 0)}</span></div>
                <div class="dl-sb-data-row"><span class="dl-sb-data-key">Targets</span><span class="dl-sb-data-val">{ds.get('unique_targets', 0)}</span></div>
                <div class="dl-sb-data-row"><span class="dl-sb-data-key">Features</span><span class="dl-sb-data-val">{metrics.get('feature_dim', 0):,}</span></div>
            </div>
        """, unsafe_allow_html=True)

        with st.expander("Model limitations"):
            st.markdown(f"""
- **Dataset**: Davis kinase binding affinity ({ds.get('unique_drugs', 0)} unique drugs, {ds.get('unique_targets', 0)} targets)
- **Task**: Binary classification (binds / doesn't bind) — no affinity regression
- **Probabilities are not calibrated** — use for ranking, not as true likelihoods
- **No 3D / docking information** — predictions use sequence and fingerprint features only
- **Out-of-domain targets** (non-kinase) are exploratory — predictions may be unreliable
- **All candidates require wet-lab experimental validation**
            """)

        st.markdown(f"""
            <div class="dl-author">
                <div class="dl-author-name">Messiah Godfred Majid</div>
                <div class="dl-author-info">
                    University of Miami<br/>
                    Computer Science · Mathematics · Biology
                </div>
                <div style="margin-top: 0.6rem;">
                    <a href="https://github.com/messiahmajid">GitHub</a> ·
                    <a href="https://linkedin.com/in/messiahmajid">LinkedIn</a> ·
                    <a href="https://messiahmajid.dev">Portfolio</a>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # ── Header ─────────────────────────────────────
    st.markdown('<div class="dl-label">Compound Screening</div>', unsafe_allow_html=True)
    st.markdown('<div class="dl-title" style="font-size: 2rem; margin-bottom: 0.25rem;">DrugLens Screening Studio</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="font-size: 0.9rem; color: var(--text-muted); margin-bottom: 1.5rem;">'
        'Interpretable compound screening for early-stage kinase drug discovery. '
        'Select a target, provide compounds, and screen for predicted binding activity.'
        '</p>',
        unsafe_allow_html=True,
    )

    # ── Compound Input ─────────────────────────────
    st.markdown('<div class="dl-label">Input</div>', unsafe_allow_html=True)
    st.markdown('<div class="dl-title">Provide compounds</div>', unsafe_allow_html=True)

    source_tab = st.radio(
        "Compound source",
        ["Example library", "Paste SMILES", "Upload CSV"],
        horizontal=True,
        label_visibility="collapsed",
        key="_compound_source",
    )

    compounds_raw: list[dict] = []
    invalid_rows: list[dict] = []

    if source_tab == "Example library":
        davis_compounds = load_davis_compounds()
        example_mode = st.radio(
            "Example set",
            ["Curated examples", "Davis training ligands"],
            horizontal=True,
            key="_example_mode",
            help="Curated examples are named common molecules. Davis ligands are the 68 local compounds used by the benchmark dataset.",
        )
        available_examples = EXAMPLE_COMPOUNDS if example_mode == "Curated examples" else davis_compounds
        if not available_examples:
            available_examples = EXAMPLE_COMPOUNDS
            st.warning("Davis ligand file was not found, so curated examples are shown instead.")

        default_count = min(10 if example_mode == "Curated examples" else 25, len(available_examples))
        selected_examples = st.multiselect(
            "Select example compounds",
            list(available_examples.keys()),
            default=list(available_examples.keys())[:default_count],
            key="_selected_examples",
            help="Choose any number of compounds from the selected example set.",
        )
        st.caption(f"{len(selected_examples)} selected from {len(available_examples)} available compounds.")
        for name in selected_examples:
            smi = available_examples[name]
            canon = canonicalize_smiles(smi)
            if canon:
                compounds_raw.append({"name": name, "smiles": smi, "canonical_smiles": canon})

    elif source_tab == "Paste SMILES":
        smiles_text = st.text_area(
            "Enter SMILES (one per line, optionally tab/space separated with name)",
            height=150,
            placeholder="CC(=O)OC1=CC=CC=C1C(=O)O\nErlotinib\tCOCCOC1=CC2=C(C=C1OCCOC)C(=NC=N2)NC3=CC=CC(=C3)C#C",
        )
        if smiles_text.strip():
            compounds_raw, invalid_rows = parse_smiles_lines(smiles_text)
            if not compounds_raw and not invalid_rows:
                st.warning("No valid SMILES found in the input.")
            elif not compounds_raw and invalid_rows:
                st.warning("No valid SMILES found. See invalid rows below.")

    elif source_tab == "Upload CSV":
        sample_csv_path = Path("examples/sample_compounds.csv")
        if sample_csv_path.exists():
            st.download_button(
                "Download sample CSV format",
                data=sample_csv_path.read_text(),
                file_name="sample_compounds.csv",
                mime="text/csv",
            )
        uploaded_file = st.file_uploader(
            "Upload CSV with a `smiles` column (optional `name` column)",
            type=["csv"],
        )
        if uploaded_file is not None:
            compounds_raw, invalid_rows = parse_compound_csv(uploaded_file)
            if not compounds_raw and not invalid_rows:
                st.warning("The uploaded CSV is empty or could not be parsed.")

    # ── Deduplicate ──
    if compounds_raw:
        pre_dedup = len(compounds_raw)
        compounds_raw = deduplicate_compounds(compounds_raw)
        n_dupes = pre_dedup - len(compounds_raw)

        col_info1, col_info2, col_info3 = st.columns(3)
        col_info1.metric("Valid compounds", len(compounds_raw))
        if n_dupes:
            col_info2.metric("Duplicates removed", n_dupes)
        if invalid_rows:
            col_info3.metric("Invalid rows", len(invalid_rows))

    # Show invalid rows
    if invalid_rows:
        with st.expander(f"⚠ {len(invalid_rows)} invalid rows", expanded=False):
            inv_df = summarize_invalid_compounds(invalid_rows)
            st.dataframe(inv_df, use_container_width=True)

    # ── Clear stale results when compound input changes ──
    _input_key = (
        source_tab,
        tuple(sorted(
            (c.get("name", ""), c.get("canonical_smiles", ""))
            for c in compounds_raw
        )),
    )
    if ("results_df" in st.session_state
            and st.session_state.get("_input_key") != _input_key):
        for key in ["results_df", "target_name", "target_info", "screening_time", "_input_key"]:
            st.session_state.pop(key, None)

    # ── Run Screening ──────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    _MAX_BATCH = 500
    if len(compounds_raw) > _MAX_BATCH:
        st.error(f"**{len(compounds_raw)} compounds** exceeds the {_MAX_BATCH}-compound limit. Only the first {_MAX_BATCH} will be screened.")
        compounds_raw = compounds_raw[:_MAX_BATCH]
    elif len(compounds_raw) > 50:
        st.warning(f"**{len(compounds_raw)} compounds** selected — screening may take a while.")

    can_screen = len(compounds_raw) > 0

    screen_button = st.button(
        f"Screen {len(compounds_raw)} compound{'s' if len(compounds_raw) != 1 else ''} against {target_choice.split(' — ')[0]}"
        if can_screen
        else "Add compounds to screen",
        type="primary",
        use_container_width=True,
        disabled=not can_screen,
    )

    if screen_button and can_screen:
        progress = st.progress(0, text="Screening compounds...")

        def update_progress(current, total):
            progress.progress(current / total, text=f"Screening compound {current}/{total}...")

        results_df = screen_compounds(
            compounds=compounds_raw,
            target_sequence=target_info["sequence"],
            model=model,
            ref_db=ref_db,
            feature_names=feature_names,
            progress_callback=update_progress,
            is_kinase=_is_kinase,
        )
        progress.empty()

        if results_df.empty:
            st.warning("No compounds could be screened. Check your input SMILES.")
        else:
            st.session_state["results_df"] = results_df
            st.session_state["target_name"] = target_choice
            st.session_state["target_info"] = target_info
            st.session_state["screening_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            st.session_state["_input_key"] = _input_key
            n_failed = len(compounds_raw) - len(results_df)
            if n_failed > 0:
                st.info(f"{n_failed} compound(s) could not be screened (featurization or prediction failure).")

    # ── Results ────────────────────────────────────
    if "results_df" in st.session_state:
        results_df = st.session_state["results_df"]
        target_name = st.session_state.get("target_name", "Unknown")
        target_info_stored = st.session_state.get("target_info", target_info)

        st.markdown("---")
        st.markdown('<div class="dl-label">Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="dl-title">Screening results</div>', unsafe_allow_html=True)

        # ── Apply filters ──
        filtered_df = results_df.copy()
        filtered_df = filtered_df[filtered_df["Binding Prob"] >= min_binding]
        if max_lipinski < 4:
            filtered_df = filtered_df[
                filtered_df["Lipinski Violations"].fillna(99).astype(int) <= max_lipinski
            ]
        filtered_df = filtered_df[
            (filtered_df["MW"].fillna(0) >= mw_range[0])
            & (filtered_df["MW"].fillna(0) <= mw_range[1])
        ]
        filtered_df = filtered_df[
            (filtered_df["LogP"].fillna(0) >= logp_range[0])
            & (filtered_df["LogP"].fillna(0) <= logp_range[1])
        ]

        # ── Summary metrics ──
        n_filtered_out = len(results_df) - len(filtered_df)
        n_high = (filtered_df["Priority"] == "High priority").sum() if not filtered_df.empty else 0
        n_review = (filtered_df["Priority"] == "Review").sum() if not filtered_df.empty else 0
        median_prob = filtered_df["Binding Prob"].median() if not filtered_df.empty else 0.0
        n_lipinski_clean = (filtered_df["Lipinski Violations"].fillna(99).astype(int) == 0).sum() if not filtered_df.empty else 0
        top_compound = filtered_df.iloc[0]["Name"] if not filtered_df.empty else "—"

        st.markdown(f"""
            <div class="dl-card" style="margin-bottom: 1rem; padding: 1rem 1.25rem;">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; color: var(--accent); text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.75rem;">Screening Summary</div>
                <div style="font-family: 'Fraunces', serif; font-size: 1.1rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.5rem;">
                    Top hit: {top_compound}
                </div>
            </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Screened", len(results_df))
        m2.metric("After filters", len(filtered_df))
        m3.metric("Filtered out", n_filtered_out)
        m4.metric("High priority", n_high)
        m5.metric("Median prob", f"{median_prob:.3f}")
        m6.metric("Lipinski clean", n_lipinski_clean)

        if filtered_df.empty:
            st.warning("No compounds match the current filters. Try adjusting filter values in the sidebar.")
        else:
            # ── Display table ──
            display_cols = [
                "Name", "SMILES", "Binding Prob", "Priority", "Trust",
                "Lipinski Violations", "MW", "LogP", "TPSA",
                "Nearest Known Compound", "Nearest Known Label",
                "Similar Known Score",
            ]
            available_cols = [c for c in display_cols if c in filtered_df.columns]
            st.dataframe(
                filtered_df[available_cols],
                use_container_width=True,
                height=min(400, 35 * len(filtered_df) + 38),
                column_config={
                    "Binding Prob": st.column_config.ProgressColumn(
                        "Binding Prob",
                        min_value=0.0,
                        max_value=1.0,
                        format="%.4f",
                    ),
                    "SMILES": st.column_config.TextColumn("SMILES", width="medium"),
                },
            )

            # ── Export CSV ──
            export_cols = [c for c in results_df.columns]
            export_df = filtered_df[export_cols].copy()

            # Add metadata as comment rows
            screening_time = st.session_state.get("screening_time", "")
            csv_buffer = io.StringIO()
            csv_buffer.write(f"# DrugLens Screening Studio — Export\n")
            csv_buffer.write(f"# Target: {target_name}\n")
            csv_buffer.write(f"# Date: {screening_time}\n")
            csv_buffer.write(f"# Model AUROC: {metrics['auroc']:.4f}, F1: {metrics['f1']:.4f}\n")
            csv_buffer.write(f"# Dataset: Davis (TDC), {ds.get('total_pairs', 0)} pairs\n")
            csv_buffer.write(f"# Compounds screened: {len(results_df)}, shown: {len(filtered_df)}\n")
            csv_buffer.write(f"# NOTE: Predictions are computational prioritization only, not experimental validation.\n")
            export_df.to_csv(csv_buffer, index=True)

            export_col1, export_col2 = st.columns(2)
            with export_col1:
                st.download_button(
                    "Download results as CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"druglens_screening_{target_name.split(' — ')[0].replace(' ', '_')}_{screening_time.replace(' ', '_').replace(':', '')}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with export_col2:
                html_report = generate_html_report(
                    filtered_df=filtered_df,
                    results_df=results_df,
                    target_name=target_name,
                    target_info=target_info_stored,
                    metrics=metrics,
                    screening_time=screening_time,
                    is_kinase=target_info_stored.get("domain") == "kinase",
                )
                st.download_button(
                    "Download HTML report",
                    data=html_report,
                    file_name=f"druglens_report_{target_name.split(' — ')[0].replace(' ', '_')}_{screening_time.replace(' ', '_').replace(':', '')}.html",
                    mime="text/html",
                    use_container_width=True,
                )

        # ── Compound Detail View ───────────────────
        st.markdown("---")
        st.markdown('<div class="dl-label">Detail</div>', unsafe_allow_html=True)
        st.markdown('<div class="dl-title">Compound evidence</div>', unsafe_allow_html=True)

        compound_options = filtered_df["Name"].tolist() if not filtered_df.empty else []
        if not compound_options:
            st.info("No compounds available for inspection with current filters.")
        else:
            selected_compound = st.selectbox(
                "Select a compound to inspect",
                compound_options,
                label_visibility="collapsed",
            )

            if selected_compound:
                row = filtered_df[filtered_df["Name"] == selected_compound].iloc[0]
                smiles = row["SMILES"]
                binding_prob = row["Binding Prob"]
                priority = row["Priority"]
                trust_level = row.get("Trust", "Medium")
                trust_reason = row.get("Trust Reason", "")

                detail_left, detail_right = st.columns([1, 2])

                with detail_left:
                    # Molecule image
                    img = smiles_to_image(smiles, size=(400, 300))
                    if img:
                        st.markdown('<div class="dl-mol-wrap">', unsafe_allow_html=True)
                        st.image(img, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Score card
                    if binding_prob >= 0.5:
                        result_class = "dl-result-binding"
                        eyebrow_class = "dl-result-eyebrow-bind"
                        verdict_class = "dl-result-verdict-bind"
                        num_class = "dl-result-num-bind"
                        verdict_text = "Likely to bind"
                    else:
                        result_class = "dl-result-nonbinding"
                        eyebrow_class = "dl-result-eyebrow-nobind"
                        verdict_class = "dl-result-verdict-nobind"
                        num_class = "dl-result-num-nobind"
                        verdict_text = "Unlikely to bind"

                    st.markdown(f"""
                        <div class="dl-result {result_class}" style="margin-top: 0.75rem;">
                            <div class="dl-result-eyebrow {eyebrow_class}">Prediction</div>
                            <div class="dl-result-verdict {verdict_class}">{verdict_text}</div>
                            <div class="dl-result-num {num_class}">{binding_prob:.1%}</div>
                            <div class="dl-result-sub">binding probability · {priority_badge_html(priority)}</div>
                        </div>
                    """, unsafe_allow_html=True)

                    # Trust card
                    trust_colors = {"High": "var(--green)", "Medium": "var(--yellow)", "Low": "var(--red)"}
                    trust_bgs = {"High": "var(--green-bg)", "Medium": "var(--yellow-bg)", "Low": "var(--red-bg)"}
                    t_color = trust_colors.get(trust_level, "var(--text-muted)")
                    t_bg = trust_bgs.get(trust_level, "var(--cream)")
                    st.markdown(f"""
                        <div style="background: {t_bg}; border: 1px solid {t_color}25; border-radius: 8px; padding: 0.75rem 1rem; margin-top: 0.6rem;">
                            <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; text-transform: uppercase; letter-spacing: 0.15em; color: {t_color};">Prediction confidence</div>
                            <div style="font-family: 'Fraunces', serif; font-size: 1.1rem; font-weight: 600; color: {t_color}; margin: 0.2rem 0;">{trust_level}</div>
                            <div style="font-family: 'Source Sans 3', sans-serif; font-size: 0.8rem; color: var(--text-secondary);">{html.escape(trust_reason)}</div>
                        </div>
                    """, unsafe_allow_html=True)

                with detail_right:
                    # ── Chemistry descriptors ──
                    desc = compute_descriptors(smiles)
                    if desc:
                        lipinski_v = count_lipinski_violations(desc)
                        st.markdown("**Chemistry descriptors**")
                        desc_col1, desc_col2, desc_col3, desc_col4 = st.columns(4)
                        desc_col1.markdown(f"""
                            <div class="dl-stat"><div class="dl-stat-val">{desc['mw']}</div><div class="dl-stat-lbl">MW</div></div>
                        """, unsafe_allow_html=True)
                        desc_col2.markdown(f"""
                            <div class="dl-stat"><div class="dl-stat-val">{desc['logp']}</div><div class="dl-stat-lbl">LogP</div></div>
                        """, unsafe_allow_html=True)
                        desc_col3.markdown(f"""
                            <div class="dl-stat"><div class="dl-stat-val">{desc['tpsa']}</div><div class="dl-stat-lbl">TPSA</div></div>
                        """, unsafe_allow_html=True)
                        desc_col4.markdown(f"""
                            <div class="dl-stat"><div class="dl-stat-val">{lipinski_v}</div><div class="dl-stat-lbl">Lipinski viol.</div></div>
                        """, unsafe_allow_html=True)

                        dc1, dc2, dc3, dc4 = st.columns(4)
                        dc1.markdown(f'<div class="dl-stat"><div class="dl-stat-val">{desc["hbd"]}</div><div class="dl-stat-lbl">HB donors</div></div>', unsafe_allow_html=True)
                        dc2.markdown(f'<div class="dl-stat"><div class="dl-stat-val">{desc["hba"]}</div><div class="dl-stat-lbl">HB acceptors</div></div>', unsafe_allow_html=True)
                        dc3.markdown(f'<div class="dl-stat"><div class="dl-stat-val">{desc["rotatable_bonds"]}</div><div class="dl-stat-lbl">Rot. bonds</div></div>', unsafe_allow_html=True)
                        dc4.markdown(f'<div class="dl-stat"><div class="dl-stat-val">{desc["aromatic_rings"]}</div><div class="dl-stat-lbl">Arom. rings</div></div>', unsafe_allow_html=True)

                    # ── Similar known compounds ──
                    st.markdown("**Similar known compounds**")
                    try:
                        similar = find_similar_drugs(
                            query_smiles=smiles,
                            reference_smiles=ref_db["smiles"],
                            reference_fingerprints=ref_db["fingerprints"],
                            reference_labels=ref_db["labels"],
                            top_k=5,
                        )
                    except Exception:
                        similar = []

                    if similar:
                        for drug in similar[:3]:
                            color = "var(--green)" if drug.get("binds") else "var(--red)"
                            status = "Known binder" if drug.get("binds") else "Non-binder"
                            dot = "●" if drug.get("binds") else "○"
                            st.markdown(f"""
                                <div class="dl-sim">
                                    <div class="dl-sim-score">Tanimoto: {drug['similarity']:.3f}</div>
                                    <div class="dl-sim-status" style="color: {color};">{dot} {status}</div>
                                    <div class="dl-sim-smiles">{drug['smiles'][:80]}{'...' if len(drug['smiles']) > 80 else ''}</div>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.caption("No similar compounds found in the reference database.")

                # ── SHAP explanation ──
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="dl-label">Explainability</div>', unsafe_allow_html=True)
                st.markdown('<div class="dl-title">What drove this prediction?</div>', unsafe_allow_html=True)
                st.markdown("""
                    <p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">
                        SHAP values show each feature's contribution.
                        <span style="color: var(--green);">Green</span> = favors binding.
                        <span style="color: var(--red);">Terracotta</span> = opposes binding.
                    </p>
                """, unsafe_allow_html=True)

                # Compute SHAP for the selected compound
                try:
                    features = featurize_pair(smiles, target_info_stored["sequence"])
                    if features is not None:
                        explainer = get_shap_explainer(model)
                        explanation = explain_prediction(explainer, features, feature_names)

                        import matplotlib as mpl
                        mpl.rcParams.update({
                            'figure.facecolor': '#faf8f5',
                            'axes.facecolor': '#faf8f5',
                            'text.color': '#2d2a26',
                            'axes.labelcolor': '#8a8078',
                            'xtick.color': '#8a8078',
                            'ytick.color': '#5c574f',
                        })

                        fig = plot_shap_bar(explanation["shap_values"], feature_names, top_k=12)
                        fig.patch.set_facecolor('#faf8f5')
                        for ax in fig.get_axes():
                            ax.set_facecolor('#faf8f5')
                        st.pyplot(fig)

                        fc1, fc2 = st.columns(2)
                        with fc1:
                            items = ''.join(
                                f'<div class="dl-feat-row"><span class="dl-feat-name">{clean_feature_name(i["feature"])}</span><span class="dl-feat-pos">+{i["shap_value"]:.4f}</span></div>'
                                for i in explanation["top_positive"][:6]
                            )
                            st.markdown(f"""
                                <div class="dl-card">
                                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; color: var(--green); text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.6rem;">Favoring binding</div>
                                    {items}
                                </div>
                            """, unsafe_allow_html=True)

                        with fc2:
                            items = ''.join(
                                f'<div class="dl-feat-row"><span class="dl-feat-name">{clean_feature_name(i["feature"])}</span><span class="dl-feat-neg">{i["shap_value"]:.4f}</span></div>'
                                for i in explanation["top_negative"][:6]
                            )
                            st.markdown(f"""
                                <div class="dl-card">
                                    <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; color: var(--red); text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.6rem;">Opposing binding</div>
                                    {items}
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("Could not compute SHAP explanation for this compound.")
                except Exception as e:
                    st.warning(f"SHAP explanation unavailable: {e}")

        # ── Caveat ─────────────────────────────────
        st.markdown("---")
        st.markdown(
            '<p style="font-size: 0.75rem; color: var(--text-muted); text-align: center; '
            'font-family: \'Source Sans 3\', sans-serif; max-width: 700px; margin: 1rem auto;">'
            '⚠ DrugLens predictions are computational prioritization based on machine learning models '
            'trained on the Davis kinase dataset. They do not constitute experimental validation, '
            'medical advice, or clinical evidence. All candidate compounds require laboratory verification.</p>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
