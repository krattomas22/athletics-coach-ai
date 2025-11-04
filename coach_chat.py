import os, io, zipfile, json, hashlib
from datetime import date
from typing import List, Dict, Any

import streamlit as st
import requests
from pypdf import PdfReader
from PIL import Image
import pytesseract

# --- Embeddings & Vector DB ---
import faiss
from sentence_transformers import SentenceTransformer

# --- LLM (OpenAI jako příklad, můžeš vyměnit) ---
from openai import OpenAI

# ========== KONFIG ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "PASTE_YOUR_KEY")  # nastav ve Streamlit Secrets pro sdílení

DEFAULT_CITY = "České Budějovice"
MODEL_EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # česky OK
MODEL_CHAT = "gpt-4o-mini"  # libovolný kompatibilní model

APP_TITLE = "Tréninkový plánovač"
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")  # sem dej svá PDF/ZIP se zdroji

# ========== UI HLAVIČKA ==========
st.set_page_config(page_title=APP_TITLE, page_icon="🏃", layout="wide")
st.title(APP_TITLE)

# ========== STAV A POMOCNÉ ==========
if "docs" not in st.session_state:
    st.session_state.docs = []  # list[dict]: {id, text, meta}
if "index" not in st.session_state:
    st.session_state.index = None
if "emb_model" not in st.session_state:
    st.session_state.emb_model = None  # lazy-load
if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=OPENAI_API_KEY)
if "assets_loaded" not in st.session_state:
    st.session_state.assets_loaded = False

def hash_text(t: str) -> str:
    return hashlib.sha1(t.encode("utf-8")).hexdigest()[:10]

def clean_text(t: str) -> str:
    return " ".join(t.replace("\n", " ").split())

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += (chunk_size - overlap)
    return chunks

def embed_texts(texts: List[str]):
    if st.session_state.emb_model is None:
        with st.spinner("Načítám embedding model…"):
            st.session_state.emb_model = SentenceTransformer(MODEL_EMB)
    return st.session_state.emb_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

def ensure_faiss(index_dim: int):
    if st.session_state.index is None:
        st.session_state.index = faiss.IndexFlatIP(index_dim)

def add_to_corpus(text: str, source: str, page: int | None = None):
    text = clean_text(text)
    if not text.strip():
        return []
    chunks = chunk_text(text)
    for j, ch in enumerate(chunks):
        meta = {"source": source, "page": page, "chunk_id": j, "id": hash_text(f"{source}-{page}-{j}")}
        st.session_state.docs.append({"id": meta["id"], "text": ch, "meta": meta})

def build_or_update_index():
    texts = [d["text"] for d in st.session_state.docs]
    if not texts:
        return
    vecs = embed_texts(texts)
    ensure_faiss(vecs.shape[1])
    st.session_state.index.reset()
    st.session_state.index.add(vecs)

def search_similar(query: str, k: int = 5) -> List[Dict[str, Any]]:
    if st.session_state.index is None or len(st.session_state.docs) == 0:
        return []
    qv = embed_texts([query])
    D, I = st.session_state.index.search(qv, k)
    out = []
    for idx in I[0]:
        if idx == -1:
            continue
        out.append(st.session_state.docs[idx])
    return out

# ========== INGEST: PDF / ZIP z assets ==========
def is_image_name(n: str) -> bool:
    n = n.lower()
    return n.endswith((".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"))

def ingest_pdf_path(path: str, label: str):
    try:
        with open(path, "rb") as f:
            reader = PdfReader(f)
            for i, page in enumerate(reader.pages):
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                add_to_corpus(txt, source=f"PDF:{os.path.basename(label)}", page=i+1)
        return True
    except Exception as e:
        st.warning(f"PDF nelze načíst ({path}): {e}")
        return False

def ingest_zip_path(path: str, label: str):
    try:
        with zipfile.ZipFile(path, "r") as z:
            names = [n for n in z.namelist() if is_image_name(n)]
            if not names:
                st.info(f"V ZIPu {label} nejsou obrázky.")
                return True
            for n in names:
                with z.open(n) as f:
                    img = Image.open(io.BytesIO(f.read())).convert("RGB")
                    txt = pytesseract.image_to_string(img, lang="ces")
                    add_to_corpus(txt, source=f"ZIP:{label}/{n}", page=None)
        return True
    except Exception as e:
        st.warning(f"ZIP nelze načíst ({path}): {e}")
        return False

def load_assets_once():
    if st.session_state.assets_loaded:
        return
    loaded_any = False
    if os.path.isdir(ASSETS_DIR):
        # Načti všechna PDF
        for name in os.listdir(ASSETS_DIR):
            p = os.path.join(ASSETS_DIR, name)
            if name.lower().endswith(".pdf"):
                ok = ingest_pdf_path(p, name)
                loaded_any = loaded_any or ok
            elif name.lower().endswith(".zip"):
                ok = ingest_zip_path(p, name)
                loaded_any = loaded_any or ok
    if loaded_any:
        build_or_update_index()
        st.session_state.assets_loaded = True

# ========== POČASÍ (wttr.in bez klíče) ==========
def get_weather(city: str = DEFAULT_CITY) -> dict:
    import urllib.parse
    city_encoded = urllib.parse.quote(city)
    url = f"https://wttr.in/{city_encoded}?format=j1"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()

        current = data["current_condition"][0]
        temp = float(current.get("temp_C", 0))
        desc = current["weatherDesc"][0]["value"]
        wind = float(current.get("windspeedKmph", 0))
        precip = float(current.get("precipMM", 0))

        return {"city": city, "temp": temp, "desc": desc, "wind": wind, "precip": precip > 0, "raw": data}
    except Exception:
        return {"city": city, "temp": 10, "desc": "nelze zjistit (offline data)", "wind": 0, "precip": False, "raw": {}}

def weather_context(w: Dict[str, Any]) -> str:
    wind_ms = w["wind"] / 3.6 if isinstance(w["wind"], (int, float)) else 0.0
    cold = (w["temp"] is not None) and (w["temp"] <= 5)
    windy = wind_ms >= 8.0  # ~28.8 km/h
    wet = bool(w["precip"])
    return "indoor" if cold or windy or wet else "outdoor"

# ========== DETERMINISTICKÝ PLÁNOVAČ ==========
def periodization(sessions_per_week: int, age: str) -> Dict[str, Any]:
    # jednoduché meta: intenzita dle věku + počtu jednotek
    base_int = "střední"
    if sessions_per_week <= 2:
        base_int = "nízká"
    elif sessions_per_week >= 4:
        base_int = "středně-vysoká"
    return {"sessions_per_week": sessions_per_week, "base_intensity": base_int, "age": age}

def generate_plan(age_group: str, context: str, pz: Dict[str, Any], races: List[Dict[str, Any]]) -> Dict[str, Any]:
    # objemové body dle věku
    base_points_map = {"U14 (do 13)": 40, "U16 (do 15)": 55}
    base_points = base_points_map.get(age_group, 45)

    # uprava dle počtu jednotek v týdnu (orientačně)
    spw = pz["sessions_per_week"]
    if spw <= 2:
        base_points = int(base_points * 0.8)
    elif spw >= 4:
        base_points = int(base_points * 1.1)

    warmup = [{"name": "běžecká abeceda", "duration": "8–10 min"},
              {"name": "mobilita kotník/kyčle", "duration": "6 min"}]
    if context == "indoor":
        main = [{"name": "6×50–60 m technický sprint 85–90 %", "rest": "90 s"},
                {"name": "koordinační žebřík", "duration": "8 min"}]
    else:
        main = [{"name": "6×80 m rovinky (80–90 %) s meziklusem", "rest": "120 s"},
                {"name": "štaf. předávky 4×50 m (technika)", "rest": "plná"}]

    strength = [{"name": "core okruh (plank, hollow, side) 2×", "duration": "10 min"}]
    cooldown = [{"name": "vyklus + strečink", "duration": "8 min"}]

    return {
        "goal": "rychlost + technika sprintu",
        "intensity": pz["base_intensity"],
        "volume_points": base_points,
        "context": context,
        "sessions_per_week": spw,
        "blocks": [
            {"part": "Warm-up", "items": warmup},
            {"part": "Main", "items": main},
            {"part": "Strength", "items": strength},
            {"part": "Cool-down", "items": cooldown},
        ],
        "safety": [
            "Nepřetěžovat nad 90 % u 11–15 let.",
            "Plná regenerace mezi opakováními.",
            "V chladnu prodloužit rozcvičení."
        ],
        "races_hint": races[:3] if races else []
    }

# ========== PROMPTY ==========
SYS_RAG = """Jsi asistent trenéra atletiky pro děti 11–15 let. Odpovídej stručně, česky.
Vycházej primárně z poskytnutých výňatků (CONTEXT). Když si nejsi jistý, řekni to.
Dbej na bezpečnost, techniku a věková omezení. Připojuj stručné reference (zdroj+strana/soubor)."""

USR_RAG = """DOTAZ: {q}
CONTEXT:
{ctx}
POKYN: Odpověz výhradně na základě CONTEXTU. Pokud něco není ve zdrojích, řekni to.
"""

SYS_PLAN = """Jsi trenér, který přetaví strukturovaný plán (JSON) do čitelného tréninku pro děti 11–15 let.
Dodrž intenzitu a objem. Nabídni 1 indoor/outdoor alternativu, pokud kontext nedává smysl.
Nakonec přidej krátké 'Proč takto' a 'Bezpečnost'. Piš česky a stručně."""

USR_PLAN = """ZÁKLAD:
{base}

METADATA:
- Věk: {age}
- Město/počasí: {city_desc}
- Počet tréninků v týdnu: {spw}

POKYN:
Sepiš 1 tréninkovou jednotku (rozcvičení → hlavní část → doplňky → cool-down),
zachovej názvy a parametry. Přidej 1 alternativu (indoor/outdoor).
"""

# ========== LEVÝ PANEL – INFO O ZDROJÍCH ==========
st.sidebar.header("📚 Zdroje")
st.sidebar.success("Zdroje jsou **přednačtené** ze složky `assets/` (PDF/ZIP).")
if st.sidebar.button("🔎 Znovu vybuildit index"):
    st.session_state.assets_loaded = False
    load_assets_once()
    st.sidebar.success("Index připraven ✅")

# ========== PRAVÝ PANEL – NASTAVENÍ ==========
st.sidebar.header("⚙️ Nastavení plánu")
age_group = st.sidebar.selectbox("Věk/skupina", ["U14 (do 13)", "U16 (do 15)"], index=0)
sessions_per_week = st.sidebar.number_input("Počet tréninků v týdnu", min_value=1, max_value=7, value=3)
city = st.sidebar.text_input("Město (počasí)", value=DEFAULT_CITY)

# Kalendář závodů – DOCX upload (volitelně)
import docx
def parse_races_docx(file_obj) -> List[Dict[str, Any]]:
    """Čeká řádky typu: 2025-11-22: 60m, dálka"""
    races = []
    try:
        doc = docx.Document(file_obj)
        for p in doc.paragraphs:
            line = p.text.strip()
            if not line:
                continue
            if ":" in line:
                date_str, rest = line.split(":", 1)
                date_str = date_str.strip()
                discs = [d.strip() for d in rest.split(",") if d.strip()]
                races.append({"date": date_str, "disciplines": discs})
    except Exception:
        pass
    return races

st.sidebar.markdown("**Kalendář závodů (volitelné, DOCX)** – každý řádek `YYYY-MM-DD: 60m, dálka`")
races_docx = st.sidebar.file_uploader("Nahrát DOCX", type=["docx"])
if races_docx:
    races = parse_races_docx(races_docx)
else:
    races = [{"date": "2025-11-22", "disciplines": ["60m", "dálka"]}]

# ========== HLAVNÍ – CHAT a PLÁN ==========
# Načti assets a index jednou
load_assets_once()

col1, col2 = st.columns([2,1])

with col1:
    st.subheader("💬 Chat nad tvými (přednačtenými) zdroji")
    q = st.text_input("Zeptej se na cokoliv z metodiky…", placeholder="Např. Jak progresovat sprinty u U14 v zimě?")
    if st.button("Odeslat dotaz") and q.strip():
        if st.session_state.index is None:
            st.warning("Zdroje nejsou načtené – klikni na 'Znovu vybuildit index'.")
        else:
            topk = search_similar(q, k=6)
            ctx_blocks = []
            for d in topk:
                meta = d["meta"]
                ref = f'{meta["source"]}{f" s.{meta["page"]}" if meta["page"] else ""}'
                ctx_blocks.append(f"[{ref}] {d['text'][:800]}")
            prompt = USR_RAG.format(q=q, ctx="\n\n".join(ctx_blocks))
            client = st.session_state.openai_client
            resp = client.chat.completions.create(
                model=MODEL_CHAT,
                messages=[{"role":"system","content":SYS_RAG},
                          {"role":"user","content":prompt}],
                temperature=0.2,
            )
            st.markdown(resp.choices[0].message.content)

with col2:
    st.subheader("🌦️ Počasí & plán")
    # počasí
    w = get_weather(city)
    st.metric("Teplota", f"{w['temp']} °C")
    wind_ms = w['wind']/3.6 if isinstance(w['wind'], (int,float)) else 0.0
    st.caption(f"{w['city']}: {w['desc']} | vítr {wind_ms:.1f} m/s")
    st.markdown("[🌦 Zobrazit radar na pocasiaradar.cz](https://www.pocasiaradar.cz/)")
    ctx = weather_context(w)

    # periodizace a plán
    pz = periodization(sessions_per_week, age_group)
    base_plan = generate_plan(age_group, ctx, pz, races)

    st.json(base_plan, expanded=False)

    if st.button("🧠 Vygenerovat čitelnou verzi"):
        client = st.session_state.openai_client
        city_desc = f"{w['city']}: {w['desc']} ({w['temp']} °C)"
        prompt = USR_PLAN.format(
            base=json.dumps(base_plan, ensure_ascii=False, indent=2),
            age=age_group,
            city_desc=city_desc,
            spw=pz["sessions_per_week"],
        )
        resp = client.chat.completions.create(
            model=MODEL_CHAT,
            messages=[{"role":"system","content":SYS_PLAN},
                      {"role":"user","content":prompt}],
            temperature=0.3,
        )
        st.markdown(resp.choices[0].message.content)

    st.download_button(
        "⬇️ Stáhnout plán (JSON)",
        data=json.dumps(base_plan, ensure_ascii=False, indent=2),
        file_name=f"plan_{date.today().isoformat()}.json",
        mime="application/json"
    )
