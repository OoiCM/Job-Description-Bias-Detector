# Dashboard.py
# Run: python -m streamlit run Dashboard.py

import os, re, string
from pathlib import Path

import numpy as np
import streamlit as st
import joblib
import pandas as pd
import html

# NEW: OCR deps
import cv2
import pytesseract
from pytesseract import Output

# text libs
import nltk
from langdetect import detect
import emoji
import spacy

# try enchant (system dep); app works without it
try:
    import enchant
    ENCHANT_OK = True
    US_DICT = enchant.Dict("en_US")
    UK_DICT = enchant.Dict("en_GB")
except Exception:
    ENCHANT_OK = False
    US_DICT = UK_DICT = None

# ---------------------------
# App config
# ---------------------------
st.set_page_config(page_title="Job Description Bias Detector (SVM) & Text OCR Extraction", page_icon="🧭", layout="wide")

# ---------------------------
# Global Style (new colorful theme)
# ---------------------------
st.markdown(
    """
    <style>
        :root{
          --bg: #0b1220;
          --card: rgba(255,255,255,.08);
          --muted: rgba(255,255,255,.65);
          --text: #eaf2ff;
          --ring: #7dd3fc;
          --gradA:#06b6d4; /* cyan-500 */
          --gradB:#8b5cf6; /* violet-500 */
          --gradC:#f43f5e; /* rose-500 */
          --chip:#0b132b;
        }

        /* page background */
        .main, .block-container {background: radial-gradient(1200px 600px at 15% 0%, rgba(99,102,241,.20), transparent 45%),
                                 radial-gradient(800px 400px at 85% 10%, rgba(6,182,212,.18), transparent 50%)
                                 , linear-gradient(180deg, #0b1220 0%, #0b1220 100%);
        }
        .block-container {max-width: 1180px; padding-top: 1.25rem;}

        /* hero header */
        .hero {
            padding: 1.4rem 1.25rem;
            border-radius: 18px;
            background: linear-gradient(135deg, var(--gradA) 0%, var(--gradB) 55%, var(--gradC) 100%);
            color: white;
            box-shadow: 0 16px 40px rgba(0,0,0,.25);
        }
        .hero h1 {margin: 0 0 .25rem 0; font-size: 1.95rem; letter-spacing:.2px}
        .hero p {margin: 0; opacity: .98}

        /* cards */
        .card {
            border-radius: 14px;
            padding: 1.1rem 1rem;
            border: 1px solid rgba(255,255,255,.08);
            background: var(--card);
            backdrop-filter: blur(10px);
            color: var(--text);
        }

        /* tabs (horizontal nav) */
        div.stTabs [data-baseweb="tab-list"]{
            gap: .35rem;
            padding: .35rem;
            border-radius: 12px;
            background: rgba(255,255,255,.05);
            border: 1px solid rgba(255,255,255,.08);
        }
        div.stTabs [data-baseweb="tab"]{
            height:auto;
            padding: .45rem .8rem;
            border-radius: 10px !important;
            background: transparent;
            color: var(--text);
            border: 1px solid transparent;
        }
        div.stTabs [aria-selected="true"]{
            background: linear-gradient(135deg, rgba(125,211,252,.20), rgba(139,92,246,.18));
            border-color: rgba(125,211,252,.35);
            box-shadow: inset 0 0 0 1px rgba(255,255,255,.08);
        }

        /* buttons */
        .stButton>button {
            border-radius: 10px;
            background: linear-gradient(90deg, var(--gradA), var(--gradB));
            color: #ffffff;
            border: 0;
            padding: .6rem 1rem;
            box-shadow: 0 8px 24px rgba(8,145,178,.35);
        }
        .stButton>button:hover {filter: brightness(1.06)}

        /* metrics and chips */
        .metric-card {display:flex; gap:.75rem; align-items:center;}
        .chip {
            display:inline-block; padding:.32rem .6rem; border-radius:9999px;
            background: linear-gradient(90deg, rgba(6,182,212,.16), rgba(139,92,246,.16));
            border: 1px solid rgba(125,211,252,.28);
            color: var(--text);
            font-size:.85rem;
        }

        /* code */
        pre, code {font-size:.92rem;}

        /* footer */
        .site-footer {margin-top: 2rem; opacity:.8; color: var(--muted)}
    </style>
    """,
    unsafe_allow_html=True,
)

ART_DIR = Path("Artifacts")
SVM_DIR = Path("SVM")
PIPE_PATH = SVM_DIR / "svm_text_pipeline.pkl"     # TF-IDF + SVM (probability=True)
LE_PATH   = ART_DIR / "label_encoder.pkl"         # class ↔ index map

# ============================
# ---- JD OCR PIPELINE ----
# ============================
OCR_MAX_SIDE        = 1800
OCR_DEFAULT_LANGS   = "eng"
OCR_DEFAULT_PSMS    = [6, 4, 11]
OCR_TIMEOUT_SEC     = 12

def _ocr_resize_cap(bgr, max_side=OCR_MAX_SIDE):
    h, w = bgr.shape[:2]
    if max(h, w) <= max_side: return bgr
    s = max_side / float(max(h, w))
    return cv2.resize(bgr, (int(w*s), int(h*s)), interpolation=cv2.INTER_AREA)

def _ocr_deskew_osd(bgr):
    """Use Tesseract OSD to determine rotation; robust to background banding."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    try:
        osd = pytesseract.image_to_osd(gray, output_type=Output.DICT)
        rot = int(osd.get("rotate", 0)) % 360
        print(f"Detected rotation angle: {rot}°")  # Debugging print
    except Exception:
        rot = 0
    
    # Prevent rotating by 180 degrees in case of false positives
    if rot == 90:
        return cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 270:
        return cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rot == 180:
        print("Warning: Image might have been incorrectly rotated by 180° based on OSD data. Skipping rotation.")
        return bgr  # Don't rotate if it's detected as 180°
    
    return bgr  # No rotation needed

def _ocr_clahe(gray, clip=1.2, tiles=(8,8)):
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=tiles).apply(gray)

def _ocr_illumination_flatten(gray, ksize=41, sigma=2, strength=1.08):
    bg   = cv2.medianBlur(gray, ksize)
    div  = cv2.divide(gray, bg, scale=255)
    fine = cv2.GaussianBlur(div, (0,0), sigma)
    return cv2.addWeighted(div, strength, fine, 1.0-strength, 0)

def _ocr_unsharp_mask(gray, sigma=1.0, amount=0.6):
    blur  = cv2.GaussianBlur(gray, (0,0), sigma)
    return cv2.addWeighted(gray, 1+amount, blur, -amount, 0)

def _ocr_remove_long_lines(gray, axis="both", max_thickness=6, min_frac=0.30):
    h, w = gray.shape
    mask = np.zeros_like(gray, np.uint8)

    def add_mask(klist, kshape, len_ok, thick_ok):
        nonlocal mask
        for k in klist:
            kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, kshape(k))
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            _, bw    = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            num, lab, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
            for i in range(1, num):
                x, y, ww, hh, _a = stats[i]
                if len_ok(ww, hh) and thick_ok(ww, hh):
                    mask[lab == i] = 255

    if axis in ("h", "both"):
        add_mask([21, 31, 41, 51],
                 kshape=lambda k: (k, 1),
                 len_ok=lambda ww, hh: ww >= int(min_frac * w),
                 thick_ok=lambda ww, hh: hh <= max_thickness)

    if axis in ("v", "both"):
        add_mask([21, 31, 41, 51],
                 kshape=lambda k: (1, k),
                 len_ok=lambda ww, hh: hh >= int(min_frac * h),
                 thick_ok=lambda ww, hh: ww <= max_thickness)

    return cv2.inpaint(gray, mask, 3, cv2.INPAINT_TELEA) if np.any(mask) else gray

def preprocess_for_tesseract(bgr, max_thickness=6, min_frac=0.30,
                             do_unsharp=True, binarize_mode="None"):
    bgr = _ocr_resize_cap(bgr, OCR_MAX_SIDE)
    bgr = _ocr_deskew_osd(bgr)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=6)

    flat = _ocr_illumination_flatten(gray, ksize=41, sigma=2, strength=1.08)
    noline = _ocr_remove_long_lines(flat, axis="both", max_thickness=max_thickness, min_frac=min_frac)

    contrast = _ocr_clahe(noline, clip=1.2, tiles=(8,8))
    if do_unsharp:
        contrast = _ocr_unsharp_mask(contrast, sigma=1.0, amount=0.6)

    final = contrast
    mode = binarize_mode.lower()
    if mode == "otsu (global)".lower():
        _, final = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "adaptive (gaussian)".lower():
        final = cv2.adaptiveThreshold(contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 35, 10)
    elif mode == "sauvola".lower():
        try:
            from skimage.filters import threshold_sauvola
            thr = threshold_sauvola(contrast, window_size=37, k=0.18)
            final = ((contrast > thr).astype(np.uint8) * 255)
        except Exception:
            _, final = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dbg = {"flat": flat, "noline": noline, "contrast": contrast}
    return final, dbg

def _ocr_mean_conf_from_dict(d):
    if not d or "conf" not in d:
        return 0.0
    vals = [int(c) for c in d["conf"] if c not in ("-1", "", None)]
    return float(np.clip(np.mean(vals), 0.0, 100.0)) if len(vals) else 0.0

def run_tesseract(img, langs="eng", psm_list=(6,4,11), timeout=12):
    best = {"conf": -1.0, "psm": None, "text": ""}
    rows = []
    for psm in psm_list:
        cfg = f"--oem 1 --psm {psm}"
        try:
            data = pytesseract.image_to_data(img, lang=langs, config=cfg,
                                             output_type=Output.DICT, timeout=timeout)
            text = pytesseract.image_to_string(img, lang=langs, config=cfg, timeout=timeout)
        except RuntimeError:
            rows.append({"psm": psm, "mean_conf": float("nan")})
            continue
        conf = _ocr_mean_conf_from_dict(data)
        rows.append({"psm": psm, "mean_conf": conf})
        if conf > best["conf"]:
            best = {"conf": conf, "psm": psm, "text": text}
    table = pd.DataFrame(rows).sort_values("mean_conf", ascending=False, ignore_index=True)
    return best["text"], best["conf"], best["psm"], table

# ---------------------------
# NLTK resources
# ---------------------------
def _nltk_dl(pkg):
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg.split("/")[-1])

_nltk_dl("corpora/stopwords")
_nltk_dl("corpora/wordnet")
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words("english"))
CUSTOM_STOPWORDS = {
    "job","role","position","responsibility","responsibilities","duty","duties",
    "requirement","requirements","description","descriptions","report","reports",
    "reporting","team","teams","work","working","department","departments",
    "company","companies","location","locations","benefit","benefits",
    "compensation","salary","experienced","year","years","including","regard"
}

# ---------------------------
# spaCy
# ---------------------------
@st.cache_resource
def load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        from spacy.cli import download
        download("en_core_web_sm")
        return spacy.load("en_core_web_sm")

nlp = load_spacy()

# ---------------------------
# Load artifacts (cache)
# ---------------------------
@st.cache_resource
def load_artifacts():
    pipe = joblib.load(PIPE_PATH)
    le   = joblib.load(LE_PATH)
    return pipe, le

try:
    PIPE, LE = load_artifacts()
    CLASS_NAMES = list(LE.classes_)
except Exception as e:
    st.error(f" Failed to load artifacts. Expected:\n- {PIPE_PATH}\n- {LE_PATH}\n\n{e}")
    st.stop()

# ---------------------------
# Regex + cleaners (TEXT)
# ---------------------------
URL_RE   = re.compile(r"https?://\S+|www\.\S+|[a-zA-Z0-9.-]+\.com\b")
HTML_RE  = re.compile(r"<.*?>")
EMAIL_RE = re.compile(r"\S+@\S+")
PUNCT    = string.punctuation + "–—"
PUNCT_RE = re.compile(f"[{re.escape(PUNCT)}]")
DIGIT_RE = re.compile(r"\d+")

def remove_urls(t):       return URL_RE.sub("", t)
def remove_html(t):       return HTML_RE.sub("", t)
def remove_emails(t):     return EMAIL_RE.sub("", t)
def remove_newlines(t):   return t.replace("\n", " ")
def remove_emoji(t):      return emoji.replace_emoji(t, replace="")
def remove_non_english_text(t):
    try:
        return t if detect(t) == "en" else ""
    except Exception:
        return t
def remove_punct(t):      return PUNCT_RE.sub(" ", t)
def remove_digits(t):     return DIGIT_RE.sub("", t)

# Contractions
CONTRACTIONS = {
    **{
        "ain't": "is not", "aren't": "are not", "can't": "cannot", "couldn't": "could not",
        "didn't": "did not", "doesn't": "does not", "don't": "do not",
        "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
        "isn't": "is not", "mustn't": "must not", "needn't": "need not",
        "shan't": "shall not", "shouldn't": "should not", "wasn't": "was not",
        "weren't": "were not", "won't": "will not", "wouldn't": "would not",
        "mightn't": "might not", "mayn't": "may not", "oughtn't": "ought not",
    },
    **{
        "could've": "could have", "might've": "might have", "must've": "must have",
        "should've": "should have", "would've": "would have", "he'd": "he would",
        "he'll": "he will", "he's": "he is", "she'd": "she would", "she'll": "she will",
        "she's": "she is", "they'd": "they would", "they'll": "they will", "they're": "they are",
        "they've": "they have", "we'd": "we would", "we're": "we are", "we've": "we have",
        "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have",
    },
    **{
        "y'all": "you all", "y'all'd": "you all would", "y'all're": "you all are",
        "i'm": "i am", "i've": "i have", "i'll": "i will", "i'd": "i would",
        "'cause": "because", "ma'am": "madam", "let's": "let us", "o'clock": "of the clock",
    },
    **{
        "i'd've": "i would have", "i'll've": "i will have", "you'd've": "you would have",
        "you'll've": "you will have", "we'd've": "we would have", "we'll've": "we will have",
        "they'd've": "they would have", "they'll've": "they will have", "she'd've": "she would have",
        "he'd've": "he would have", "it'd": "it would", "it'd've": "it would have",
        "it'll": "it will", "it'll've": "it will have", "who'll": "who will",
        "who'll've": "who will have", "who've": "who have", "who's": "who is",
        "what'll": "what will", "what'll've": "what will have", "what're": "what are",
        "what's": "what is", "what've": "what have", "where'd": "where did",
        "where's": "where is", "where've": "where have", "when's": "when is",
        "when've": "when have", "why's": "why is", "why've": "why have",
        "that'd": "that would", "that'd've": "that would have", "that's": "that is",
        "there'd": "there would", "there'd've": "there would have", "there's": "there is",
        "here's": "here is", "how'd": "how did", "how'd'y": "how do you",
        "how'll": "how will", "how's": "how is", "to've": "to have", "will've": "will have",
        "so's": "so as", "so've": "so have", "this's": "this is", "sha'n't": "shall not",
        "mustn't've": "must not have", "mightn't've": "might not have", "shouldn't've": "should not have",
        "wouldn't've": "would not have", "y'all'd've": "you all would have",
    },
    **{
        "u.s": "america", "e.g": "for example"
    }
}

def expand_contractions(text:str)->str:
    for s in ["’","‘","´","`"]:
        text = text.replace(s, "'")
    for c, e in CONTRACTIONS.items():
        text = re.sub(rf"\b{re.escape(c)}\b", e, text)  # <- fixed
    return text

def tokens_en_filter(tokens):
    if not ENCHANT_OK:
        return tokens
    keep = []
    for tok in tokens:
        if US_DICT.check(tok) or UK_DICT.check(tok) or (tok.isalpha() and len(tok) > 2):
            keep.append(tok)
    return keep

def preprocess_text(raw: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        return ""

    t = raw.lower()
    t = remove_urls(t)
    t = remove_html(t)
    t = remove_newlines(t)
    t = remove_punct(t)
    t = remove_emails(t)
    t = remove_emoji(t)
    t = remove_non_english_text(t)
    t = remove_digits(t)

    t = expand_contractions(t)

    doc = nlp(t)

    toks = []
    for token in doc:
        lemma = token.lemma_.strip()
        if not lemma or not lemma.isalpha():
            continue
        lw = lemma.lower()
        if lw in STOPWORDS or lw in CUSTOM_STOPWORDS:
            continue
        toks.append(lw)

    toks = tokens_en_filter(toks)
    return " ".join(toks)

# =========================
# Bias dictionary utilities
# =========================
DICT_PATHS = [
    Path("Artifacts") / "bias_word_dictionaries_with_severity.xlsx",
    Path("bias_word_dictionaries_with_severity.xlsx"),
]

TERM_CANDIDATES = ["term", "word", "words", "phrase", "bias_word", "keyword", "pattern"]
CAT_CANDIDATES  = ["category", "bias_category", "label", "type", "group"]
SEV_CANDIDATES  = ["severity", "weight", "score", "level", "priority"]

@st.cache_resource
def load_bias_lexicon():
    path = None
    for p in DICT_PATHS:
        if p.exists():
            path = p
            break
    if path is None:
        return None

    try:
        xls = pd.ExcelFile(path)
        frames = [xls.parse(s) for s in xls.sheet_names]
        raw = pd.concat(frames, ignore_index=True)
    except Exception as e:
        st.warning(f"Could not read dictionary Excel at {path}: {e}")
        return None

    cols_lower = {c.lower().strip(): c for c in raw.columns}
    
    # Select the correct columns for term, subtype, and severity
    term_col = cols_lower.get("term") or cols_lower.get("word")  # Term column
    subtype_col = cols_lower.get("subtype") or cols_lower.get("category")  # Category column (subtype)
    sev_col = cols_lower.get("severity") or cols_lower.get("weight")  # Severity column

    if not term_col or not subtype_col:
        st.warning("Bias dictionary: Missing term, category (subtype), or severity columns.")
        return None
    
    # Process the data into a DataFrame
    df = pd.DataFrame({"term": raw[term_col].astype(str).str.strip().str.lower()})
    df["subtype"] = raw[subtype_col].astype(str).str.strip()  # Correct the column name here
    df["severity"] = raw[sev_col] if sev_col else pd.NA
    df = df[df["term"] != ""].drop_duplicates(subset=["term", "subtype", "severity"])

    # Generate patterns for each term to search for in text
    def term_to_pattern(t):
        parts = [re.escape(p) for p in t.split()]
        if not parts: return None
        sep = r"(?:\s|[-_/])+"  # To handle spaces and certain separators
        pattern = r"\b" + sep.join(parts) + r"\b"
        return re.compile(pattern, flags=re.IGNORECASE)

    df["pattern"] = df["term"].apply(term_to_pattern)
    df = df[df["pattern"].notna()]
    return df

LEXICON = load_bias_lexicon()

# Now update the 'find_bias_matches' function to handle 'subtype' column
def find_bias_matches(original_text: str, lexicon: pd.DataFrame):
    if lexicon is None or not isinstance(original_text, str) or not original_text.strip():
        return []
    matches = []
    for row in lexicon.itertuples(index=False):
        pat = row.pattern
        for m in pat.finditer(original_text):
            matches.append({
                "start": m.start(),
                "end": m.end(),
                "matched": original_text[m.start():m.end()],
                "term": row.term,
                "category": row.subtype,  # Use 'subtype' instead of 'category'
                "severity": row.severity,
            })
    matches.sort(key=lambda d: (d["start"], -(d["end"]-d["start"])))
    merged, last_end = [], -1
    for m in matches:
        if m["start"] >= last_end:
            merged.append(m)
            last_end = m["end"]
    return merged

PALETTE = [
    "#fde68a", "#fca5a5", "#93c5fd", "#86efac", "#f0abfc",
    "#fda4af", "#a5b4fc", "#34d399", "#fcd34d", "#c7d2fe"
]
def color_for_category(cat: str):
    if not hasattr(color_for_category, "_map"):
        color_for_category._map = {}
    cmap = color_for_category._map
    if cat not in cmap:
        cmap[cat] = PALETTE[len(cmap) % len(PALETTE)]
    return cmap[cat]

def highlight_bias_html(original_text: str, matches: list):
    if not matches:
        return html.escape(original_text)
    out, cursor = [], 0
    for m in matches:
        out.append(html.escape(original_text[cursor:m["start"]]))
        span = html.escape(original_text[m["start"]:m["end"]])
        cat  = html.escape(str(m["category"]))
        term = html.escape(str(m.get("term", "")))
        color = color_for_category(str(m["category"]))
        out.append(
            f'<mark style="background:{color}; padding:0 .2rem; border-radius:4px" '
            f'title="{cat}{": " + term if term else ""}">{span}</mark>'
        )
        cursor = m["end"]
    out.append(html.escape(original_text[cursor:]))
    return "".join(out)

def aggregate_matches(matches: list) -> pd.DataFrame:
    if not matches:
        return pd.DataFrame(columns=["term","category","severity","count"])
    dfm = pd.DataFrame(matches)
    agg = (dfm
           .groupby(["term","category","severity"], dropna=False)
           .agg(count=("matched","count"))
           .sort_values("count", ascending=False)
           .reset_index())
    return agg

def build_bias_legend_html(cat_counts):
    """Return HTML for the category/color legend with NO leading spaces/newlines."""
    if cat_counts is None or len(cat_counts) == 0:
        return ""
    items = []
    for cat, cnt in cat_counts.items():
        color = color_for_category(str(cat))
        items.append(
            # single-line, no indentation so Markdown doesn't treat as code block
            f"<div style='display:flex;align-items:center;gap:.5rem;padding:.25rem .5rem;margin:.15rem .4rem;"
            f"background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.08);border-radius:8px;'>"
            f"<span style='width:14px;height:14px;border-radius:4px;background:{color};"
            f"box-shadow:inset 0 0 0 1px rgba(0,0,0,.25);'></span>"
            f"<span style='color:var(--text)'><b>{html.escape(str(cat))}</b></span>"
            f"<span style='opacity:.75'>× {int(cnt)}</span>"
            f"</div>"
        )
    return (
        "<div style='display:flex;flex-wrap:wrap;align-items:center;'>"
        + "".join(items) +
        "</div>"
    )

# ---------------------------
# HERO
# ---------------------------
with st.container():
    st.markdown(
        """
        <div class="hero">
            <h1>🧭 Job Description Bias Detector (SVM) & 📄 Text OCR Extraction </h1>
            <p>Analyze bias in Job Description with the SVM model or extract Job Description text from images using a robust Tesseract pipeline.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================
# Top Horizontal Navigation (Tabs)
# ============================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["OCR Extractor", "Detector","Bias Word Dictionary", "How it works", "About"])

# ============================
# TAB 1: OCR Extractor
# ============================
with tab1:
    st.write("")  # tiny spacer

    # Layout: left = upload/result; right = settings
    left, right = st.columns([1.2, 0.8], gap="large")

    with left:
        st.markdown("### Upload & Extract")
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload a JD image",
            type=["png","jpg","jpeg","tif","tiff","bmp","webp"],
            help="One page image works best."
        )

        # Auto-run toggle (default ON)
        st.checkbox("Auto-run when a file is uploaded", value=True, key="_autorun_ocr")

        # Track new file selection to allow auto-run once per new file
        current_name = uploaded.name if uploaded is not None else None
        last_name = st.session_state.get("_last_upload_name")
        if current_name and current_name != last_name:
            st.session_state["_ocr_autorun_consumed"] = False
            st.session_state["_last_upload_name"] = current_name

        run_ocr_click = st.button("🖨️ Extract text")

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if 'last_ocr' in st.session_state:
            c1, c2 = st.columns([1,1])
            c1.markdown(
                f"<div class='metric-card'><span class='chip'>Best PSM</span><h4>{st.session_state['last_ocr'].get('psm')}</h4></div>",
                unsafe_allow_html=True
            )
            c2.markdown(
                f"<div class='metric-card'><span class='chip'>Mean confidence</span><h4>{st.session_state['last_ocr'].get('conf'):.2f}</h4></div>",
                unsafe_allow_html=True
            )
        else:
            st.caption("OCR results will appear here after you extract text.")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("### OCR Settings")
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        with st.expander("Tesseract path (Windows)"):
            tess_path = st.text_input(
                "Path to tesseract.exe (optional)",
                value="",
                help="Set if Tesseract isn't on PATH, e.g. C:\\\\Program Files\\\\Tesseract-OCR\\\\tesseract.exe"
            )
            if tess_path and os.path.exists(tess_path):
                pytesseract.pytesseract.tesseract_cmd = tess_path

        psm_opts  = st.multiselect("PSMs to try", options=list(range(0,14)), default=OCR_DEFAULT_PSMS)
        timeout   = st.slider("Per-PSM timeout (s)", 5, 30, OCR_TIMEOUT_SEC, 1)

        st.markdown("---")
        st.markdown("#### Preprocessing")
        max_thickness = st.slider("Max line thickness (px)", 2, 12, 6, 1)
        min_frac      = st.slider("Min line length (fraction of page)", 0.10, 0.60, 0.30, 0.05)
        binarize      = st.selectbox("Binarization", ["None", "Otsu (global)", "Adaptive (Gaussian)", "Sauvola"], index=0)
        use_unsharp   = st.checkbox("Tiny unsharp mask (clarity)", value=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Auto-run orchestration ----------
    autorun_enabled = st.session_state.get("_autorun_ocr", True)
    autorun_ready = (uploaded is not None) and autorun_enabled and (not st.session_state.get("_ocr_autorun_consumed", False))
    run_ocr = bool(run_ocr_click)
    if autorun_ready:
        run_ocr = True
        st.session_state["_ocr_autorun_consumed"] = True

    # ---------- Run OCR when requested ----------
    if run_ocr:
        if uploaded is None:
            st.warning("Please upload an image.")
        else:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if bgr is None:
                st.error("Could not read the image. Please try another file.")
                st.stop()

            # Status bar around heavy steps
            with st.status("Running OCR pipeline...", expanded=False) as status:
                status.update(label="Preprocessing image…", state="running")
                final_img, dbg = preprocess_for_tesseract(
                    bgr,
                    max_thickness=max_thickness,
                    min_frac=min_frac,
                    do_unsharp=use_unsharp,
                    binarize_mode=binarize
                )

                status.update(label="Trying PSMs with Tesseract…", state="running")
                text, mean_conf, best_psm, table = run_tesseract(
                    final_img, langs="eng", psm_list=psm_opts, timeout=timeout
                )
                status.update(label="Done ✅", state="complete")

            # Visuals + results
            c1, c2, c3, c4 = st.columns(4)
            c1.image(cv2.cvtColor(_ocr_resize_cap(bgr), cv2.COLOR_BGR2RGB), caption="Original")
            c2.image(dbg["flat"], caption="Flatten", clamp=True)
            c3.image(dbg["noline"], caption="No lines (H+V)", clamp=True)
            c4.image(final_img, caption=f"To Tesseract (BIN={binarize})", clamp=True)

            # Store OCR text ONLY in a buffer; do NOT auto-copy to detector
            st.session_state['last_ocr'] = {"psm": best_psm, "conf": mean_conf}
            st.session_state['latest_jd_text'] = text

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"**Best PSM:** `{best_psm}` &nbsp;&nbsp; **Mean conf:** `{mean_conf:.2f}`")
            st.dataframe(table, use_container_width=True)
            st.subheader("Extracted Job Description")
            st.text_area("Text", text, height=300, key="jd_text_out")

            colA, colB, colC = st.columns([1,1,1])
            with colA:
                st.download_button(
                    "Download text",
                    data=text.encode("utf-8"),
                    file_name=f"{os.path.splitext(uploaded.name)[0]}.tess.txt",
                    mime="text/plain",
                )
            with colB:
                # DO NOT set st.session_state["text_area"] here anymore
                st.info("Review the extracted text. When ready, click **Send to Detector** below.")
            with colC:
                st.info("Then open the **Detector** tab to classify the text.")
            st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Send to Detector only on button click ----------
    have_text = bool(st.session_state.get("latest_jd_text", "").strip())
    if st.button("Send to Detector (open the Detector tab)", disabled=not have_text):
        st.session_state["text_area"] = st.session_state["latest_jd_text"]
        st.success("Sent to Detector. Switch to the **Detector** tab to run prediction.")
        # Optional: st.rerun() if you want the Detector tab to immediately see the updated state
        # st.rerun()

    # --- Help / Explainer for OCR settings (keep your existing blocks below) ---
    st.markdown("### What do these settings do?")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    # ... your existing explanation content ...
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("#### Quick tips")
    st.markdown(
        """
    - If output looks **broken or missing lines** → try **PSM 11** (sparse text).  
    - If **table/grid lines** appear in OCR → raise **Max line thickness** a little.  
    - If **letters are getting cut** → lower **Max line thickness** or raise **Min line length**.  
    - If background is **uneven** → try **Adaptive (Gaussian)** or **Sauvola** binarization.  
    - If it’s **slow** on big images** → reduce **timeout** or resize source images before upload.
    """
    )

    st.markdown("### Performance tips")
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(
        f"""
    - **Image resolution**: Prefer **150–300 dpi**. Very large images are capped to **{OCR_MAX_SIDE}px** on the long side to keep latency predictable.
    - **File format**: Use lossless or lightly-compressed images (`.png`, high-quality `.jpg`). Heavy JPEG artifacts hurt OCR.
    - **Layout**: For **single block** layouts start with **PSM 6**. For **single column** with varied fonts try **PSM 4**. For messy or cropped content try **PSM 11**.
    - **Lines & rules**: If tables/underlines leak into text, increase **Max line thickness**; if real letters get erased, decrease it or raise **Min line length**.
    - **Contrast**: Low-contrast scans benefit from **Adaptive (Gaussian)** or **Sauvola** binarization; otherwise keep **None** (grayscale often works best).
    - **Timeouts**: If you hit timeouts, reduce the **PSM list** or lower the **per-PSM timeout** (current default: **{OCR_TIMEOUT_SEC}s**).
    - **Language packs**: Install the correct Tesseract language data and set `langs="eng"` (or more) if you expand beyond English later.
    - **Windows path**: If Tesseract isn’t on PATH, set the `tesseract.exe` path in the expander on the right.
    - **Quality check**: Use the **mean confidence table** to guide tweaks; re-run with different PSMs/binarization when confidence is low.
    """
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ============================
# TAB 2: Detector (clean, no Explanation)
# ============================
with tab2:
    st.write("")
    with st.expander("About preprocessing", expanded=False):
        st.write(
            "- Removes URLs, HTML, emails, newlines, emoji, non-English, punctuation, digits\n"
            "- Expands contractions\n"
            "- spaCy tokenization → lemmatization\n"
            "- Dictionary filter (pyenchant) when available\n"
            "- Produces the same **lemmatized tokens joined** representation used for TF-IDF"
        )

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown("### Input")
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        # Initialize the text_area with a default value if not set yet
        if "text_area" not in st.session_state:
            st.session_state["text_area"] = ""

        text = st.text_area("Paste a job description:", key="text_area", height=260, placeholder="Paste or type here…")

        ca, cb = st.columns([1, 1])
        with ca:
            # Button sets a trigger so we still run after rerun resets the UI
            if st.button("🔮 Predict", disabled=not bool(text.strip()), key="predict_button"):
                st.session_state["predict_trigger"] = True
                st.session_state["predict_text"] = text

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("### Result")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if 'last_pred' in st.session_state:
            st.metric("Bias category", st.session_state['last_pred'])
        else:
            st.caption("Prediction will appear here after you click Predict.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Run prediction if trigger is set (survives the rerun)
    if st.session_state.get("predict_trigger"):
        text_to_use = st.session_state.get("predict_text", "")
        if not text_to_use.strip():
            st.warning("Please enter some text.")
        else:
            cleaned = preprocess_text(text_to_use)
            if not cleaned:
                st.warning("After preprocessing the text is empty (likely non-English or only symbols).")
            else:
                pred_idx = PIPE.predict([cleaned])[0]
                pred_label = LE.inverse_transform([pred_idx])[0]
                try:
                    proba = PIPE.predict_proba([cleaned])[0]
                except Exception:
                    proba = None

                st.session_state['last_pred'] = pred_label

                # Tabs (no Explanation)
                tabs = st.tabs(["Prediction", "Preprocessed", "Confidence", "Bias words"])

                with tabs[0]:
                    st.markdown("#### Prediction")
                    st.markdown(f"<span class='chip'>Bias category</span> &nbsp; **{pred_label}**", unsafe_allow_html=True)

                with tabs[1]:
                    st.markdown("#### Preprocessed text")
                    st.code(cleaned[:1500] + ("..." if len(cleaned) > 1500 else ""))

                with tabs[2]:
                    if proba is not None:
                        dfp = pd.DataFrame({"Class": LE.classes_, "Probability": proba}).sort_values("Probability", ascending=False)
                        st.bar_chart(dfp.set_index("Class"))
                        st.caption("Higher bar → higher model confidence for the class.")
                        st.dataframe(dfp.reset_index(drop=True), use_container_width=True)
                    else:
                        st.info("Confidence chart unavailable (model is not probability-enabled).")

                with tabs[3]:
                    st.markdown("#### Dictionary matches")
                    if LEXICON is None:
                        st.info("Bias dictionary not found. Place it at `Artifacts/bias_word_dictionaries_with_severity.xlsx` or update `DICT_PATHS`.")
                    else:
                        matches = find_bias_matches(text_to_use, LEXICON)  # highlight original text (not cleaned)
                        if not matches:
                            st.success("No bias terms from the dictionary were detected in this text.")
                        else:
                            # counts per category
                            cat_counts = pd.DataFrame(matches).groupby("category").size().sort_values(ascending=False)

                            # Category color legend (render as HTML)
                            st.markdown("#### Category color legend")
                            legend_html = build_bias_legend_html(cat_counts)
                            if legend_html:
                                st.markdown(legend_html, unsafe_allow_html=True)

                            # Highlighted JD text with tooltips
                            st.markdown("#### Highlighted text")
                            st.markdown(highlight_bias_html(text_to_use, matches), unsafe_allow_html=True)

                            st.markdown("#### Matched terms")
                            st.dataframe(aggregate_matches(matches), use_container_width=True)

                # Clear trigger so it won't re-run on every rerun
                st.session_state["predict_trigger"] = False
                st.toast("Prediction complete — check the Detector tab.", icon="✅")


# ============================
# TAB 3: Bias Word Dictionary
# ============================
with tab3:  # Add this line to the list of tabs
    st.write("### Bias Dictionary")

    # Ensure lexicon is loaded
    if LEXICON is None:
        st.warning("Bias dictionary not found. Place it at `Artifacts/bias_word_dictionaries_with_severity.xlsx` or update `DICT_PATHS`.")
    else:
        # Search functionality
        search_term = st.text_input("Search for a bias word", "")

        # If the search term is provided, filter the lexicon
        if search_term:
            filtered_dict = LEXICON[LEXICON['term'].str.contains(search_term, case=False, na=False)]
        else:
            filtered_dict = LEXICON

        # Display total bias terms in dictionary
        st.markdown(f"**Total bias terms in dictionary**: {len(filtered_dict)}")

        # Show categories available in the dictionary (subtypes)
        subtypes = filtered_dict["subtype"].unique()
        subtype_filter = st.multiselect(
            "Filter by category", 
            options=subtypes.tolist(), 
            default=subtypes.tolist()
        )

        # Filter the dictionary by selected subtypes
        filtered_dict = filtered_dict[filtered_dict["subtype"].isin(subtype_filter)]

        # Show bias words and related details in a table
        st.dataframe(filtered_dict[['term', 'subtype', 'severity']].sort_values(by='subtype'))

        # Option to download the dictionary as CSV
        st.markdown("### Download Dictionary")
        st.download_button(
            label="Download CSV",
            data=filtered_dict.to_csv(index=False).encode('utf-8'),
            file_name="bias_dictionary.csv",
            mime="text/csv"
        )

        # Optional: Display a bar chart for category counts
        subtype_counts = filtered_dict['subtype'].value_counts()
        st.bar_chart(subtype_counts)


# ============================
# TAB 4: How it works
# ============================
with tab4:
    st.markdown("### How it works")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown(
        """
**End-to-end flow**

1. **Input source** → You either paste a Job Description (JD) or upload an image/PDF page of a JD.
2. **OCR (if image)** → Tesseract extracts text after robust pre-processing (deskew, illumination flatten, H/V line removal, CLAHE, optional binarization) and multi-PSM search. The best PSM is chosen by mean confidence.
3. **Text preprocessing** → Reproduces the *training-time* pipeline exactly: lowercasing, URL/HTML/email/emoji/digit/punctuation removal, contraction expansion, spaCy tokenization + lemmatization, NLTK stopwords + domain stopwords, and optional dictionary word filter (pyenchant).
4. **Vectorization & inference** → TF-IDF transforms the text into features, then an **SVM** predicts the **bias category**. If the model supports probabilities, a confidence bar chart is shown.
5. **Dictionary matching** → The JD is scanned against a bias lexicon (`term / subtype / severity`) with whitespace/sep-aware regexes; matched spans are highlighted and aggregated.
6. **Session handoff** → OCR output can be sent to the Detector tab via session state (`latest_jd_text` / `text_area`) to keep the experience fluid.
"""
    )

    with st.expander("Preprocessing details"):
        st.markdown(
            """
- **Normalization**: lowercase → contract expansions (e.g., “don’t” → “do not”).
- **Cleaning**: remove URLs, HTML, emails, emoji, digits, and punctuation; preserve word boundaries.
- **Linguistic**: spaCy **lemmatization** keeps “engineers/engineering” as *engineer*; filters NLTK stopwords plus **domain stopwords** (e.g., *role/requirement/department*).
- **Optional dictionary filter**: when **pyenchant** is present, very rare non-words are dropped to reduce noise.
- **Exact parity with training**: the Detector uses the **same** steps to avoid train/serve skew.
"""
        )

    with st.expander("OCR pipeline details"):
        st.markdown(
            """
- **OSD deskew** (Tesseract) avoids tilted pages; skips 180° flip unless confidence suggests upside-down text.
- **Illumination flattening** (median blur division) reduces shadows and background banding.
- **Rule removal**: morphological black-hat detects long horizontal/vertical lines (tables/underlines) for inpainting.
- **Contrast boost**: CLAHE (+ optional unsharp mask).
- **Binarization** *(optional)*: Otsu (global), Adaptive (Gaussian), or Sauvola for textured paper.
- **Multi-PSM search**: typically PSM 6 / 4 / 11; best chosen by mean word confidence.
- **Performance tips**: very large images are capped to a max side ({} px) to keep latency predictable; per-PSM timeout defaults to {} s.
""".format(OCR_MAX_SIDE, OCR_TIMEOUT_SEC)
        )
        st.caption("Note: set `Tesseract.exe` path on Windows if it's not on PATH.")

    with st.expander("Confidence & calibration"):
        st.markdown(
            """
- **Probabilities** come from `SVC(probability=True)` (Platt scaling inside scikit-learn). They are useful for ranking but are **not** perfectly calibrated.
- If you need calibrated scores for decisions, consider wrapping with `CalibratedClassifierCV` or using temperature scaling on a held-out set.
"""
        )

    with st.expander("Reproducibility & artifacts"):
        st.markdown(
            f"""
- **Artifacts**: `{PIPE_PATH.name}` (TF-IDF + SVM), `{LE_PATH.name}` (label encoder).
- **Classes**: {", ".join(CLASS_NAMES) if 'CLASS_NAMES' in globals() else 'loaded at runtime'}.
- **Determinism**: fixed random states in training; serving is deterministic for the same input.
- **Environment**: Python + Streamlit + scikit-learn + spaCy + NLTK + OpenCV + Tesseract.
"""
        )

    with st.expander("Privacy & security"):
        st.markdown(
            """
- Processing is **local** to your machine/session. No JD text or images are sent to external services.
- Text is stored only in Streamlit **session state** for convenience and cleared when you refresh/close.
- Exports (CSV/text) are **explicit** user actions.
"""
        )

    st.markdown("</div>", unsafe_allow_html=True)

# ============================
# TAB 5: About (final with env checker)
# ============================
with tab5:
    st.markdown("### About this app")
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown(
        """
**Purpose & intended use**

This dashboard demonstrates an applied NLP + OCR system for **detecting and quantifying bias indicators in Job Descriptions (JDs)**.  
It supports your academic project objectives (SDG 10 – Reduced Inequalities): analysis, experimentation, and stakeholder demos.

**What’s inside**

- **Bias Detector** — TF-IDF + SVM classifier trained on preprocessed JD text.
- **JD OCR** — Image-to-text extraction using Tesseract with robust pre-processing and multi-PSM search.
- **Bias Dictionary Explorer** — Search/filter a curated lexicon (`term / subtype / severity`), with in-text highlights and counts.
"""
    )

    st.markdown("#### Assumptions")
    st.markdown(
        """
- Input is **English** JD text or **printed** JD images (single page works best).
- The detector expects the **same preprocessing** used during training (this app enforces it).
- The bias lexicon is **curated/heuristic** and supplements model predictions; it is **not** a ground-truth authority.
"""
    )

    st.markdown("#### Limitations & known gaps")
    st.markdown(
        """
- **Language**: English-only; code-mixed or non-English text may be dropped by language filtering.
- **Generalization**: Model is trained on specific JD distributions; niche domains or highly creative writing may reduce accuracy.
- **Probability calibration**: SVM probabilities are approximate; avoid hard thresholds for high-stakes decisions.
- **Dictionary precision**: Regex matches can trigger **false positives** (e.g., overlapping words, context-dependent phrases).
- **OCR constraints**: Handwriting, heavy stylization, low-res scans, complex multi-column layouts, or dense tables can degrade extraction.
- **Scope**: This is an **analysis tool**; it does **not** automate hiring decisions or produce legal/fairness guarantees.
"""
    )

    st.markdown("#### Privacy")
    st.markdown(
        """
- No uploaded text/images are sent to external APIs; all processing is local.
- The app only stores content in temporary **session state** unless you explicitly download it.
"""
    )

    st.markdown("#### Reproducibility")
    st.markdown(
        f"""
- **Artifacts**: `{PIPE_PATH.name}` (pipeline) and `{LE_PATH.name}` (labels) are loaded at runtime.
- **Classes**: {", ".join(CLASS_NAMES) if 'CLASS_NAMES' in globals() else 'loaded at runtime'}.
- **Dependencies**: scikit-learn, spaCy, NLTK, OpenCV, pytesseract/Tesseract, Streamlit.
- For strict reproducibility, pin package versions and archive the artifacts with a commit hash.
"""
    )

    st.markdown("#### Roadmap ideas")
    st.markdown(
        """
- **Model**: upgrade to transformer baselines (e.g., DistilBERT/RoBERTa) + calibrated probabilities.
- **Explainability**: token/phrase attributions (LIME/SHAP) and confusion-matrix/error-analysis views.
- **Multilingual**: extend to major languages; toggle code-mixed tolerance.
- **Batching & export**: drop a folder of JDs (text/PDF/images), get a consolidated CSV report.
- **PDF multipage OCR**: page loop, optional table detection, and layout-aware reading.
- **Bias lexicon**: richer taxonomy, context rules to reduce false positives, severity normalization.
- **Packaging**: Docker image, optional REST API, and GPU support for OCR/transformers.
"""
    )

    # NEW: Environment & artifacts check
    with st.expander("Environment & artifacts check", expanded=False):
        import sklearn, platform
        st.markdown("**Versions**")
        st.code(
            f"Python: {platform.python_version()}\n"
            f"OS: {platform.system()} {platform.release()}\n"
            f"scikit-learn: {sklearn.__version__}\n"
            f"spaCy: {spacy.__version__}\n"
            f"NLTK: {nltk.__version__}\n"
            f"OpenCV: {cv2.__version__}\n"
            f"pytesseract: {pytesseract.get_tesseract_version() if hasattr(pytesseract,'get_tesseract_version') else 'unknown'}"
        )

        st.markdown("**Artifacts**")
        for p in [PIPE_PATH, LE_PATH]:
            st.write(f"- {p} — {'✅ found' if p.exists() else '❌ missing'}")

        # Optional: gentle warning if sklearn version not what you trained on
        EXPECTED_SKLEARN = "1.7.1"  # set to your training version if known
        if 'PIPE' in globals() and sklearn.__version__ != EXPECTED_SKLEARN:
            st.warning(
                f"Artifacts likely trained with scikit-learn {EXPECTED_SKLEARN}, "
                f"but runtime is {sklearn.__version__}. Align versions or re-export for exact parity."
            )

    # Model card (concise)
    with st.expander("Model card (concise)", expanded=False):
        from datetime import datetime

        st.markdown(
            """
**Intended use**  
Interactive analysis of **English** Job Descriptions to surface **bias indicators** and predict a **bias category** for research, education, and demos.

**Out-of-scope / Not for**  
Automated hiring decisions, compliance/legal determinations, or individual-level judgments. Results are **indicative**, not definitive.

**Inputs & preprocessing**  
- Inputs: UTF-8 JD text or printed JD images (OCR).  
- Preprocessing: lowercase → contractions → remove URLs/HTML/emails/emoji/digits/punct → spaCy lemmatization → stopwords + domain stopwords → optional dictionary filter (pyenchant).  
- Serving uses the **same** preprocessing as training to avoid train/serve skew.

**Model**  
- Architecture: **TF-IDF + SVM** (`probability=True`).  
- Output: single **bias category** label; probabilities are approximate (Platt scaling).  
- Classes: shown below (from loaded label encoder).

**Data considerations**  
- Trained on job-description style text; unusual domains or highly stylized writing may reduce accuracy.  
- Dictionary highlights are **pattern matches** and can produce false positives without context.

**Fairness & limitations**  
- English-only, no code-mixed support (yet).  
- OCR struggles with handwriting, heavy stylization, or complex multi-column layouts.  
- Probability calibration is imperfect; avoid hard thresholds for high-stakes use.

**Artifacts & versions**  
- Pipeline: `{PIPE_PATH.name}`  
- Labels: `{LE_PATH.name}`  
"""
        )

        # Show classes if available
        try:
            st.markdown("**Loaded classes**")
            st.code(", ".join(CLASS_NAMES))
        except Exception:
            st.info("Classes will display after artifacts load successfully.")

        # Optional: last-modified timestamps for artifacts
        rows = []
        for p in [PIPE_PATH, LE_PATH]:
            try:
                ts = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                rows.append(f"- `{p}` — last modified: {ts}")
            except Exception:
                rows.append(f"- `{p}` — not found at runtime")
        st.markdown("\n".join(rows))

    st.write(f"Dictionary filtering: {'enabled' if ENCHANT_OK else 'unavailable'}")
    st.markdown("</div>", unsafe_allow_html=True)




