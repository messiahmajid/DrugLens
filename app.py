"""
DrugLens — Streamlit Web Application
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from rdkit import Chem
from rdkit.Chem import Draw, AllChem

from src.features import featurize_pair, get_all_feature_names
from src.model import load_artifacts, predict_binding
from src.similarity import find_similar_drugs
from src.explainability import get_shap_explainer, explain_prediction, plot_shap_bar


st.set_page_config(
    page_title="DrugLens",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Anthropic-Inspired CSS ───────────────────────────────────────────────

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
    }

    .stApp {
        background: var(--cream) !important;
    }

    .stApp > header { background: transparent !important; }

    .main .block-container {
        padding-top: 2rem !important;
        max-width: 1100px !important;
    }

    /* ── Sidebar ──────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--warm-white) !important;
        border-right: 1px solid var(--border) !important;
    }

    section[data-testid="stSidebar"] * {
        font-family: 'Source Sans 3', sans-serif !important;
    }

    /* ── Metrics ──────────────────────────────────── */
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

    /* ── Typography ───────────────────────────────── */
    h1, h2, h3 {
        font-family: 'Fraunces', serif !important;
        color: var(--text-primary) !important;
    }

    p, li, span, label {
        font-family: 'Source Sans 3', sans-serif !important;
        color: var(--text-secondary) !important;
    }

    /* ── Inputs ───────────────────────────────────── */
    .stSelectbox > div > div,
    .stTextInput > div > div > input {
        background: var(--warm-white) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-family: 'Source Sans 3', sans-serif !important;
    }

    .stSelectbox label, .stTextInput label {
        color: var(--text-muted) !important;
        font-size: 0.85rem !important;
    }

    /* ── Button ───────────────────────────────────── */
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

    /* ── Expander ─────────────────────────────────── */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary {
        background: var(--cream) !important;
        border-radius: 8px !important;
        font-family: 'Source Sans 3', sans-serif !important;
        color: var(--text-secondary) !important;
        padding: 0.6rem 1rem !important;
        gap: 0.5rem !important;
    }

    [data-testid="stExpander"] summary span {
        font-family: 'Source Sans 3', sans-serif !important;
        color: var(--text-secondary) !important;
        overflow: visible !important;
        white-space: nowrap !important;
    }

    [data-testid="stExpander"] summary svg {
        flex-shrink: 0 !important;
        margin-right: 0.25rem !important;
    }

    .streamlit-expanderContent,
    [data-testid="stExpander"] [data-testid="stExpanderDetails"] {
        background: var(--warm-white) !important;
        border: 1px solid var(--border-light) !important;
    }

    /* ── Code ─────────────────────────────────────── */
    code, .stCode {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
    }

    /* ── Dividers ─────────────────────────────────── */
    hr { border-color: var(--border-light) !important; }

    /* ── Alert ────────────────────────────────────── */
    .stAlert {
        background: var(--cream) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    /* ─── Custom Components ───────────────────────── */
    .dl-hero {
        text-align: center;
        padding: 0.5rem 0 2.5rem 0;
        border-bottom: 1px solid var(--border-light);
        margin-bottom: 2rem;
    }

    .dl-hero-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 0.2em;
    }

    .dl-hero-title {
        font-family: 'Fraunces', serif !important;
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        margin: 0.25rem 0 !important;
        line-height: 1.1 !important;
    }

    .dl-hero-sub {
        font-family: 'Source Sans 3', sans-serif;
        font-size: 1rem;
        color: var(--text-muted);
        font-weight: 300;
    }

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
        margin-bottom: 1rem;
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

    /* ── Prediction ───────────────────────────────── */
    .dl-result {
        border-radius: 10px;
        padding: 1.75rem;
        text-align: center;
        position: relative;
    }

    .dl-result-binding {
        background: var(--green-bg);
        border: 1px solid rgba(90, 158, 111, 0.25);
    }

    .dl-result-nonbinding {
        background: var(--red-bg);
        border: 1px solid rgba(196, 112, 90, 0.25);
    }

    .dl-result-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
    }

    .dl-result-eyebrow-bind { color: var(--green); }
    .dl-result-eyebrow-nobind { color: var(--red); }

    .dl-result-verdict {
        font-family: 'Fraunces', serif;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0.4rem 0 0.2rem 0;
    }

    .dl-result-verdict-bind { color: var(--green); }
    .dl-result-verdict-nobind { color: var(--red); }

    .dl-result-num {
        font-family: 'Fraunces', serif;
        font-size: 2.8rem;
        font-weight: 700;
        line-height: 1;
    }

    .dl-result-num-bind { color: var(--green); }
    .dl-result-num-nobind { color: var(--red); }

    .dl-result-sub {
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.8rem;
        color: var(--text-muted);
        margin-top: 0.3rem;
    }

    /* ── Stat boxes ───────────────────────────────── */
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

    /* ── Feature list ─────────────────────────────── */
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

    /* ── Similar drugs ────────────────────────────── */
    .dl-sim {
        background: var(--warm-white);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.5rem;
    }

    .dl-sim-score {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        color: var(--text-primary);
        font-weight: 500;
    }

    .dl-sim-status {
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.8rem;
    }

    .dl-sim-smiles {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        color: var(--text-muted);
        margin-top: 0.2rem;
        word-break: break-all;
    }

    /* ── Sidebar brand ────────────────────────────── */
    .dl-sb-brand {
        padding: 0.25rem 0 1.25rem 0;
        border-bottom: 1px solid var(--border-light);
        margin-bottom: 1rem;
    }

    .dl-sb-name {
        font-family: 'Fraunces', serif;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    .dl-sb-tag {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.6rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }

    /* ── Author ───────────────────────────────────── */
    .dl-author {
        background: var(--cream);
        border: 1px solid var(--border-light);
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1.5rem;
    }

    .dl-author-name {
        font-family: 'Fraunces', serif;
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--text-primary);
    }

    .dl-author-info {
        font-family: 'Source Sans 3', sans-serif;
        font-size: 0.75rem;
        color: var(--text-muted);
        line-height: 1.6;
    }

    .dl-author a {
        color: var(--accent);
        text-decoration: none;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
    }

    .dl-author a:hover { text-decoration: underline; }

    /* ── Sidebar data card ────────────────────────── */
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

    .dl-sb-data-key {
        font-family: 'Source Sans 3', sans-serif;
        color: var(--text-muted);
    }

    .dl-sb-data-val {
        font-family: 'IBM Plex Mono', monospace;
        color: var(--text-primary);
        font-weight: 500;
    }

    /* ── Hide streamlit chrome ────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─── Data ──────────────────────────────────────────────────────────────────

EXAMPLE_TARGETS = {
    "EGFR — non-small cell lung cancer": {
        "sequence": "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVISDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPSIATGMVGALLLLLVVALGIGLFMRRRHIVRKRTLRRLLQERELVEPLTPSGEAPNQALLRILKETEFKKIKVLGSGAFGTVYKGLWIPEGEKVKIPVAIKELREATSPKANKEILDEAYVMASVDNPHVCRLLGICLTSTVQLITQLMPFGCLLDYVREHKDNIGSQYLLNWCVQIAKGMNYLEDRRLVHRDLAARNVLVKTPQHVKITDFGLAKLLGAEEKEYHAEGGKVPIKWMALESILHRIYTHQSDVWSYGVTVWELMTFGSKPYDGIPASEISSILEKGERLPQPPICTIDVYMIMVKCWMIDADSRPKFRELIIEFSKMARDPQRYLVIQGDERMHLPSPTDSNFYRALMDEEDMDDVVDADEYLIPQQGFFSSPSTSRTPLLSSLSATSNNSTVACIDRNGLQSCPIKEDSFLQRYSSDPTGALTEDSIDDTFLPVPEYINQSVPKRPAGSVQNPVYHNQPLNPAPSRDPHYQDPHSTAVGNPEYLNTVQPTCVNSTFDSPAHWAQKGSHQISLDNPDYQQDFFPKEAKPNGIFKGSTAENAEYLRVAPQSSEFIGA",
        "description": "Key target in lung cancer. Known drugs: Erlotinib, Gefitinib."
    },
    "ACE2 — SARS-CoV-2 entry receptor": {
        "sequence": "MSSSSWLLLSLVAVTAAQSTIEEQAKTFLDKFNHEAEDLFYQSSLASWNYNTNITEENVQNMNNAGDKWSAFLKEQSTLAQMYPLQEIQNLTVKLQLQALQQNGSSVLSEDKSKRLNTILNTMSTIYSTGKVCNPDNPQECLLLEPGLNEIMANSLDYNERLWAWESWRSEVGKQLRPLYEEYVVLKNEMARANHYEDYGDYWRGDYEVNGVDGYDYSRGQLIEDVEHTFEEIKPLYEHLHAYVRAKLMNAYPSYISPIGCLPAHLLGDMWGRFWTNLYSLTVPFGQKPNIDVTDAMVDQAWDAQRIFKEAEKFFVSVGLPNMTQGFWENSMLTDPGNVQKAVCHPTAWDLGKGDFRILMCTKVTMDDFLTAHHEMGHIQYDMAYAAQPFLLRNGANEGFHEAVGEIMSLSAATPKHLKSIGLLSPDFQEDNETEINFLLKQALTIVGTLPFTYMLEKWRWMVFKGEIPKDQWMKKWWEMKREIVGVVEPVPHDETYCDPASLFHVSNDYSFIRYYTRTLYQFQFQEALCQAAKHEGPLHKCDISNSTEAGQKLFNMLRLGKSEPWTLALENVVGAKNMNVRPLLNYFEPLFTWLKDQNKNSFVGWSTDWSPYAD",
        "description": "COVID-19 viral entry point. Target for antiviral therapeutics."
    },
    "COX-2 — inflammation & pain": {
        "sequence": "MLARALLLCAVLALSHTANPCCSHPCQNRGVCMSVGFDQYKCDCTRTGFYGENCSTPEFLTRIKLFLKPTPNTVHYILTHFKGFWNVVNNIPFLRNAIMSYVLTSRSHLIDSPPTYNADYGYKSWEAFSNLSYYTRALPPVPDDCPTPLGVKGKKQLPDSNEIVEKLLLRRKFIPDPQGSNMMFAFFAQHFTHQFFKTDHKRGPAFTNGLGHGVDLNHIYGETLARQRKLRLFKDGKMKYQIIDGEMYPPTVKDTQAEMIYPPQVPEHLRFAVGQEVFGLVPGLMMYATIWLREHNRVCDVLKQEHPEWGDEQLFQTTRLILIGETIKIVIEDYVQHLSGYHFKLKFDPELLFNKQFQYQNRIAAEFNTLYHWHPLLPDTFQIHDQKYNYQQFIYNNSILLEHGITQFVESFTRQIAGRVAGGRNVPPAVQKVSQASIDQSRQMKYQSFNEYRKRFMLKPYESFEELTGEKEMSAELEALYGDIDAVELYPALLVEKPRPDAIFGETMVEVGAPFSLKGLMGNVICSPAYWKPSTFGGEVGFQIINTASIQSLICNNVKGCPFTSFSVPDPELIKTVTINASSSRSGLDDINPTVLLKERSTEL",
        "description": "Anti-inflammatory target. Drugs: Celecoxib, Ibuprofen."
    },
    "CDK2 — cell cycle regulation": {
        "sequence": "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLESEEEGVPSTAIREISLLKELKHDNIVRLYDIVHSDAHKLYLVFEFLDLDLKRYMEGIPKDQPLGADIVKKFMMQLCKGIAYCHSHRILHRDLKPQNLLIDKEGNLKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL",
        "description": "Cell cycle regulator. Validated cancer drug target."
    },
    "BRAF V600E — melanoma": {
        "sequence": "MEHIQGAWKTISNGFGFKDAVFDGSSCISPTIVQQFGYQRRASDDGKLTDPSKTSNTIRVFLPNKDEVVDIVPPEIKQKVLDEGVNIKKSEVKIITQIVKDINRSGIDTITSIEHQLELIQCGATTSEYIYRDEEITQIKTVLNELIPGESQIQLIEGTPYQIMSDNLRCEVNAVHPHIKRLLNDNTIAIIELPHAGISRYDSQENQHIIKSIVKFLDASHTRLAGRDLHSSLRSSFQHLTRYALPFEDKNDGKQIDISPNFHGQPITNSSPLKSRDFISNGLCIKNLENIAQVSSERNTVDITIALSGLKYNKHEFIAFDNENIPLYYYQDESIAYNHSSEDEIITPVQGSTFPAWYLNKEKLTKQEFPYVVSYMSSFSLSSIRGVDSGIMVHISVFVNKFVEKPTQHSNESFCYFLHKFLYSNNILLLHEGIRVRGEKSFMKNFETKVK",
        "description": "Melanoma oncogene. Drugs: Vemurafenib, Dabrafenib."
    },
    "DPP-4 — type 2 diabetes": {
        "sequence": "MKTPWKVLLGLLGAAALVTIITVPVVLLNKGTDDATADSRKTYTLTDYLKNTYRLKLYSLRWISDHEYLYKQENNILVFNAEYGNSSVFLENSTFDEFGHSINDYSISPDGQFILLEYNYVKQWRHSYTASYDIYDLNKRQLITEERIPNNTQWVTWSPVGHKLAYVWNNDIYVKIEPNLPSYRITWTGKEDIIYNGITDWVYEEEVFSAYSALWWSPNGTFLAYAQFNDTEVPLIEYSFYSDESLQYPKTVRVPYPKAGAVNPTVKFFVVNTDSLSSVTNATSIQITAPASMLIGDHYLCDVTWATQERISLQWLRRIQNYSVMDICDYDESSGRWNCLVARQHIEMSTTGWVGRFRPSEPHFTLDGNSFYKIISNEEGYRHICYFQIDKKDCTFITKGTWEVIGIEALTSDYLYYISNEYKGMPGGRNLYKIQLSDYTKVTCLSCELNPERCQYYSVSFSKEAKYYQLRCSGPGLPLYTLHSSVNDKGLRVLEDNSALDKMLQNVQMPSKKLDFIILNETKFWYQMILPPHFDKSKKYPLLLDVYAGPCSQKADTVFRLNWATYLASTENIIVASFDGRGSGYQGDKIMHAINRRLGTFEVEDQIEAARQFSKMGFVDNKRIAIWGWSYGGYVTSMVLGSGSGVFKCGIAVAPVSRWEYYDSVYTERYMGLPTPEDNLDHYRNSTVMSRAENFKQVEYLLIHGTADDNVHFQQSAQISKALVDVGVDFQAMWYTDEDHGIASSTAHQHIYTHMSHFIKQCFSLP",
        "description": "Type 2 diabetes target. Drugs: Sitagliptin (Januvia)."
    },
}

EXAMPLE_DRUGS = {
    "Staurosporine — broad kinase inhibitor": "O=C1NC2=CC=CC=C2C1=CC3=CN=C4C=CC=CC34",
    "Erlotinib — EGFR inhibitor (lung cancer)": "COCCOC1=CC2=C(C=C1OCCOC)C(=NC=N2)NC3=CC=CC(=C3)C#C",
    "Imatinib — BCR-ABL inhibitor (leukemia)": "CN1CCN(CC1)CC2=CC=C(C=C2)C(=O)NC3=CC(=C(C=C3)NC4=NC=CC(=N4)C5=CN=CC=C5)C",
    "Aspirin — COX inhibitor": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "Metformin — diabetes": "CN(C)C(=N)NC(=N)N",
    "Enter custom SMILES": "",
}


@st.cache_resource
def load_model():
    return load_artifacts("artifacts")

def smiles_to_image(smiles, size=(400, 300)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Draw.MolToImage(mol, size=size)

def validate_smiles(smiles):
    return Chem.MolFromSmiles(smiles) is not None

def clean_feature_name(name):
    if name.startswith("MorganBit_"):
        return f"Substructure #{name.split('_')[1]}"
    return name.replace("_", " ")


def main():
    if not Path("artifacts/model.joblib").exists():
        st.error("Model not found. Run `python train.py` first.")
        return

    model, metrics, ref_db = load_model()
    feature_names = get_all_feature_names()
    explainer = get_shap_explainer(model)

    # ── Sidebar ────────────────────────────────────
    with st.sidebar:
        st.markdown("""
            <div class="dl-sb-brand">
                <div class="dl-sb-name">DrugLens</div>
                <div class="dl-sb-tag">Drug–Target Interaction Predictor</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("### Performance")
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

    # ── Hero ───────────────────────────────────────
    st.markdown("""
        <div class="dl-hero">
            <div class="dl-hero-eyebrow">Computational Drug Discovery</div>
            <div class="dl-hero-title">DrugLens</div>
            <div class="dl-hero-sub">Predict whether a drug molecule will bind to a protein target — the core question behind every drug discovery program.</div>
        </div>
    """, unsafe_allow_html=True)

    # ── Inputs ─────────────────────────────────────
    col_d, col_t = st.columns(2)

    with col_d:
        st.markdown('<div class="dl-label">Drug</div>', unsafe_allow_html=True)
        st.markdown('<div class="dl-title">Select a molecule</div>', unsafe_allow_html=True)

        drug_choice = st.selectbox("Drug:", list(EXAMPLE_DRUGS.keys()), label_visibility="collapsed")

        if drug_choice == "Enter custom SMILES":
            smiles_input = st.text_input("SMILES:", placeholder="CC(=O)OC1=CC=CC=C1C(=O)O")
        else:
            smiles_input = EXAMPLE_DRUGS[drug_choice]
            st.code(smiles_input, language=None)

        if smiles_input and validate_smiles(smiles_input):
            img = smiles_to_image(smiles_input)
            if img:
                st.markdown('<div class="dl-mol-wrap">', unsafe_allow_html=True)
                st.image(img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
        elif smiles_input:
            st.error("Invalid SMILES string.")

    with col_t:
        st.markdown('<div class="dl-label">Target</div>', unsafe_allow_html=True)
        st.markdown('<div class="dl-title">Select a protein</div>', unsafe_allow_html=True)

        target_choice = st.selectbox("Target:", list(EXAMPLE_TARGETS.keys()), label_visibility="collapsed")
        info = EXAMPLE_TARGETS[target_choice]

        st.markdown(f"""
            <div class="dl-card">
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.6rem; color: var(--accent); text-transform: uppercase; letter-spacing: 0.15em; margin-bottom: 0.5rem;">Therapeutic context</div>
                <div style="font-family: 'Source Sans 3', sans-serif; color: var(--text-secondary); font-size: 0.9rem; line-height: 1.5;">{info['description']}</div>
                <div style="font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: var(--text-muted); margin-top: 0.6rem;">{len(info['sequence'])} amino acid residues</div>
            </div>
        """, unsafe_allow_html=True)

        with st.expander("View sequence"):
            st.code(info["sequence"][:300] + "...", language=None)

    # ── Predict ────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Predict binding interaction", type="primary", use_container_width=True):
        if not smiles_input or not validate_smiles(smiles_input):
            st.error("Enter a valid SMILES string.")
            return

        with st.spinner("Analyzing molecular interaction..."):
            features = featurize_pair(smiles_input, info["sequence"])
            if features is None:
                st.error("Could not process this molecule.")
                return

            prediction, confidence = predict_binding(model, features)
            explanation = explain_prediction(explainer, features, feature_names)
            similar = find_similar_drugs(
                query_smiles=smiles_input,
                reference_smiles=ref_db["smiles"],
                reference_fingerprints=ref_db["fingerprints"],
                reference_labels=ref_db["labels"],
                top_k=5,
            )

        # ── Results ────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="dl-label">Prediction</div>', unsafe_allow_html=True)
        st.markdown('<div class="dl-title">Binding analysis</div>', unsafe_allow_html=True)

        rc1, rc2 = st.columns([2, 1])

        with rc1:
            if prediction == 1:
                st.markdown(f"""
                    <div class="dl-result dl-result-binding">
                        <div class="dl-result-eyebrow dl-result-eyebrow-bind">Result</div>
                        <div class="dl-result-verdict dl-result-verdict-bind">Likely to bind</div>
                        <div class="dl-result-num dl-result-num-bind">{confidence:.1%}</div>
                        <div class="dl-result-sub">binding probability</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="dl-result dl-result-nonbinding">
                        <div class="dl-result-eyebrow dl-result-eyebrow-nobind">Result</div>
                        <div class="dl-result-verdict dl-result-verdict-nobind">Unlikely to bind</div>
                        <div class="dl-result-num dl-result-num-nobind">{1-confidence:.1%}</div>
                        <div class="dl-result-sub">non-binding probability</div>
                    </div>
                """, unsafe_allow_html=True)

        with rc2:
            st.markdown(f"""
                <div class="dl-stat"><div class="dl-stat-val">{confidence:.4f}</div><div class="dl-stat-lbl">Binding score</div></div>
                <div class="dl-stat"><div class="dl-stat-val">0.50</div><div class="dl-stat-lbl">Threshold</div></div>
                <div class="dl-stat"><div class="dl-stat-val">{metrics['auroc']:.3f}</div><div class="dl-stat-lbl">Model AUROC</div></div>
            """, unsafe_allow_html=True)

        # ── SHAP ───────────────────────────────────
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

        # ── Similar ────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="dl-label">Discovery</div>', unsafe_allow_html=True)
        st.markdown('<div class="dl-title">Similar known compounds</div>', unsafe_allow_html=True)
        st.markdown('<p style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 1rem;">Structurally similar molecules from the training database, ranked by Tanimoto similarity.</p>', unsafe_allow_html=True)

        if similar:
            for drug in similar:
                sc1, sc2 = st.columns([1, 3])
                with sc1:
                    sim_img = smiles_to_image(drug["smiles"], size=(200, 150))
                    if sim_img:
                        st.markdown('<div class="dl-mol-wrap">', unsafe_allow_html=True)
                        st.image(sim_img)
                        st.markdown('</div>', unsafe_allow_html=True)
                with sc2:
                    color = "var(--green)" if drug.get("binds") else "var(--red)"
                    status = "Known binder" if drug.get("binds") else "Non-binder"
                    dot = "●" if drug.get("binds") else "○"
                    st.markdown(f"""
                        <div class="dl-sim">
                            <div class="dl-sim-score">Tanimoto similarity: {drug['similarity']:.3f}</div>
                            <div class="dl-sim-status" style="color: {color};">{dot} {status}</div>
                            <div class="dl-sim-smiles">{drug['smiles'][:90]}{'...' if len(drug['smiles']) > 90 else ''}</div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No similar compounds found in the reference database.")


if __name__ == "__main__":
    main()
