import os, io, zipfile, json, hashlib
import random  # pro jitter v backoffu
from datetime import date
from typing import List, Dict, Any

import streamlit as st
import requests
from pypdf import PdfReader
from PIL import Image
import pytesseract
import time
from openai import RateLimitError, APIError

# --- Embeddings & Vector DB ---
import faiss
from sentence_transformers import SentenceTransformer

# --- LLM (OpenAI jako příklad, můžeš vyměnit) ---
from openai import OpenAI

# ========== KONFIG ==========
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "PASTE_YOUR_KEY")

DEFAULT_CITY = "České Budějovice"
MODEL_EMB = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # umí dobře česky
MODEL_CHAT = "gpt-4o-mini"  # nebo jiný dle dostupnosti
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")  # <-- sem dávej své PDF/ZIP

# ========== UI HLAVIČKA ==========
st.set_page_config(page_title="Athletics Coach AI", page_icon="🏃", layout="wide")
st.title("🏃‍♂️ Athletics Coach – RAG Chat + Tréninkový plánovač")

# ========== STAV A POMOCNÉ ==========
if "docs" not in st.session_state:
    st.session_state.docs = []  # list[dict]: {id, text, meta}
if "index" not in st.session_state:
    st.session_state.index = None
if "emb_model" not in st.session_state:
    # lazy-load kvůli paměti
    st.session_state.emb_model = None
if "openai_client" not in st.session_state:
    st.session_state.openai_client = OpenAI(api_key=OPENAI_API_KEY)

def safe_chat_completion(client, messages, model, temperature=0.2, max_retries=6):
    """
    Volá OpenAI Chat s automatickým retry při RateLimitError/APIError.
    Exponenciální backoff + náhodný jitter: 1s → 2s → 4s → 8s → 12s → 16s (+0–1s).
    """
    delay = 1.0
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
            )
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            sleep_for = delay + random.uniform(0, 1)
            st.info(f"⏳ Limit API – zkusím znovu za {sleep_for:.1f} s…")
            time.sleep(sleep_for)
            delay = min(delay * 2, 16)
        except APIError:
            if attempt == max_retries - 1:
                raise
            sleep_for = delay + random.uniform(0, 1)
            st.info("⚠️ Dočasná chyba služby – opakuji požadavek…")
            time.sleep(sleep_for)
            delay = min(delay * 2, 16)

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
        return
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

def _is_image_name(n: str) -> bool:
    n = n.lower()
    return n.endswith((".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"))

def load_assets_if_needed():
    """
    Načte všechny PDF a ZIP (s fotkami stránek) z adresáře ASSETS_DIR, jednou.
    """
    if st.session_state.assets_loaded:
        return
    if not os.path.isdir(ASSETS_DIR):
        st.warning(f"Adresář assets nenalezen: {ASSETS_DIR}")
        return

    loaded_pages = 0
    loaded_imgs = 0

    with st.spinner("Načítám zdroje z assets/…"):
        for name in sorted(os.listdir(ASSETS_DIR)):
            path = os.path.join(ASSETS_DIR, name)
            if not os.path.isfile(path):
                continue

            # PDF -> vytěžit text
            if name.lower().endswith(".pdf"):
                try:
                    reader = PdfReader(path)
                    for i, page in enumerate(reader.pages):
                        try:
                            txt = page.extract_text() or ""
                        except Exception:
                            txt = ""
                        add_to_corpus(txt, source=f"PDF:{name}", page=i+1)
                        loaded_pages += 1
                except Exception as e:
                    st.warning(f"PDF se nepodařilo načíst ({name}): {e}")

            # ZIP -> OCR z obrázků
            elif name.lower().endswith(".zip"):
                try:
                    with zipfile.ZipFile(path, "r") as z:
                        for n in z.namelist():
                            if not _is_image_name(n):
                                continue
                            with z.open(n) as f:
                                try:
                                    img = Image.open(io.BytesIO(f.read())).convert("RGB")
                                    txt = pytesseract.image_to_string(img, lang="ces")
                                except Exception:
                                    txt = ""
                                add_to_corpus(txt, source=f"ZIP:{name}/{n}", page=None)
                                loaded_imgs += 1
                except Exception as e:
                    st.warning(f"ZIP se nepodařilo načíst ({name}): {e}")

    build_or_update_index()
    st.session_state.assets_loaded = True
    st.success(f"Zdroje načteny ✅ (PDF stránek: {loaded_pages}, OCR obrázků: {loaded_imgs})")

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

# ========== INGEST: PDF / ZIP (jen z assets) ==========
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
                add_to_corpus(txt, source=f"PDF:{label}", page=i+1)
        return True
    except Exception as e:
        st.warning(f"PDF nelze načíst ({label}): {e}")
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
        st.warning(f"ZIP nelze načíst ({label}): {e}")
        return False

def load_assets_once():
    """Načte veškeré PDF/ZIP z ./assets pouze jednou a vybuduje index."""
    if st.session_state.assets_loaded:
        return
    st.session_state.docs = []
    loaded_any = False
    if os.path.isdir(ASSETS_DIR):
        for name in os.listdir(ASSETS_DIR):
            path = os.path.join(ASSETS_DIR, name)
            if name.lower().endswith(".pdf"):
                ok = ingest_pdf_path(path, name)
                loaded_any = loaded_any or ok
            elif name.lower().endswith(".zip"):
                ok = ingest_zip_path(path, name)
                loaded_any = loaded_any or ok
    if loaded_any:
        build_or_update_index()
        st.session_state.assets_loaded = True
        st.sidebar.success("Zdroje načteny z assets a index připraven ✅")
    else:
        st.sidebar.warning("Ve složce `assets/` nebyla nalezena žádná PDF/ZIP.")

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
        return {"city": city, "temp": 10, "desc": "nelze zjistit (offline)", "wind": 0, "precip": False, "raw": {}}

def weather_context(w: Dict[str, Any]) -> str:
    if w["precip"] or w["temp"] <= 5:
        return "indoor"
    return "outdoor"

# ========== DETERMINISTICKÝ PLÁNOVAČ ==========
def periodization(date_: date, season_peak: date | None, micro_week: int, age: str) -> Dict[str, Any]:
    deload = (micro_week % 4 == 0)
    base_int = "střední" if not deload else "nízká"
    return {"micro_week": micro_week, "deload": deload, "base_intensity": base_int, "age": age}

def generate_plan(age_group: str, context: str, pz: Dict[str, Any], races: List[Dict[str, Any]]) -> Dict[str, Any]:
    base_points = {"U11": 30, "U13": 40, "U15": 50}.get(age_group, 40)
    if pz["deload"]:
        base_points = int(base_points * 0.75)

    warmup = [{"name": "běžecká abeceda", "duration": "10 min"},
              {"name": "mobilita kotník/kyčle", "duration": "6 min"}]
    if context == "indoor":
        main = [{"name": "6×60 m technický sprint 85–90 %", "rest": "90 s"},
                {"name": "rychlostní žebřík – koordinace", "duration": "8 min"}]
    else:
        main = [{"name": "6×80 m rovinky (80–90 %) s meziklusem", "rest": "120 s"},
                {"name": "štafetové úseky 4×50 m (technika předávky)", "rest": "plná"}]

    strength = [{"name": "core okruh (plank, hollow, side) 2×", "duration": "10 min"}]
    cooldown = [{"name": "vyklus + strečink", "duration": "8 min"}]

    return {
        "goal": "rychlost + technika sprintu",
        "intensity": pz["base_intensity"],
        "volume_points": base_points,
        "context": context,
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
        ]
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
- Mikrocyklus: {micro_week} (deload: {deload})

POKYN:
Sepiš 1 tréninkovou jednotku (rozcvičení → hlavní část → doplňky → cool-down),
zachovej názvy a parametry. Přidej 1 alternativu (indoor/outdoor).
"""

# ========== SIDEBAR – ZDROJE (jen informace) ==========
st.sidebar.header("📚 Zdroje")
st.sidebar.info("Zdroje jsou načítány **pouze** z adresáře `assets/` v repozitáři (PDF/ZIP).")
if st.sidebar.button("🔎 Znovu načíst & vybuildit index"):
    st.session_state.assets_loaded = False
    load_assets_once()

# první načtení assets + index
load_assets_once()

# ========== PRAVÝ PANEL – NASTAVENÍ ==========
st.sidebar.header("⚙️ Nastavení plánu")
age_group = st.sidebar.selectbox("Věk/skupina", ["U11", "U13", "U15"], index=1)
micro_week = st.sidebar.number_input("Týden mikrocyklu (1–4)", min_value=1, max_value=4, value=3)
city = st.sidebar.text_input("Město (počasí)", value=DEFAULT_CITY)
# --- Nové kolonky pro nastavení tréninku ---
focus_opts = [
    "rychlost", "technika sprintu", "vytrvalost",
    "skok daleký", "skok vysoký", "vrhy/hody", "síla/CORE"
]
focuses = st.sidebar.multiselect(
    "Zaměření týdne (vyber 1–3)", focus_opts,
    default=["rychlost", "technika sprintu"]
)

sessions_per_week = st.sidebar.slider(
    "Počet tréninků v týdnu", 1, 6, 3
)

races_str = st.sidebar.text_area(
    "Kalendář závodů (JSON list)",
    value='[{"date":"2025-11-22","disciplines":["60m","dálka"]}]'
)
# Fallbacky, kdyby uživatel nic nevybral
focuses = focuses or ["rychlost"]
sessions_per_week = int(sessions_per_week or 3)

# Po kliknutí na tlačítko vygeneruj plán
if generate_clicked:
    with st.spinner("💪 Generuji plán podle nastavení..."):
        # 1) Načtení počasí a kontext (indoor/outdoor)
        w = get_weather(city)
        ctx = "indoor" if (w and w.get("indoor")) else "outdoor"

        # 2) Periodizace
        pz = periodization(date.today(), None, micro_week, age_group)

        # 3) Vygeneruj základní plán (JSON)
        try:
            base_plan = generate_plan(age_group, ctx, pz, races, focuses, sessions_per_week)
        except TypeError:
            # fallback pro starší signaturu generate_plan(age_group, ctx, pz, races)
            base_plan = generate_plan(age_group, ctx, pz, races)

        # 4) Ulož do session
        st.session_state["generated_plan"] = base_plan

        # 5) Připrav prompt a udělej čitelnou verzi (AI výstup)
        city_desc = (
            f"{w.get('city','')}: {w.get('desc','')}"
            + (f" ({w.get('temp')} °C)" if w and w.get('temp') is not None else "")
        )

        prompt = USR_PLAN.format(
            base=json.dumps(base_plan, ensure_ascii=False, indent=2),
            age=age_group,
            city_desc=city_desc,
            micro_week=pz["micro_week"],
            deload=pz["deload"],
        )

        resp = safe_chat_completion(
            client=st.session_state.openai_client,
            model=MODEL_CHAT,
            messages=[
                {"role": "system", "content": SYS_PLAN},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

    # 6) Výstup
    st.success("✅ Tréninkový plán byl úspěšně vygenerován!")
    st.markdown(resp.choices[0].message.content)

    st.download_button(
        "📥 Stáhnout plán (JSON)",
        data=json.dumps(base_plan, ensure_ascii=False, indent=2),
        file_name=f"plan_{date.today().isoformat()}.json",
        mime="application/json",
    )

# Parse závodů (bezpečně)
try:
    races = json.loads(races_str) if races_str.strip() else []
    if not isinstance(races, list):
        raise ValueError("Races must be a JSON list.")
except Exception as e:
    st.sidebar.error(f"Chyba v JSONu závodů: {e}")
    races = []

generate_clicked = st.sidebar.button("💪 Vygenerovat plán", type="primary")
# ========== HLAVNÍ – CHAT ==========
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("💬 Chat nad přednačtenými zdroji (assets)")
    q = st.text_input(
        "Zeptej se na cokoliv z metodiky…",
        placeholder="Např. Jak progresovat sprinty u U13 v zimě?"
    )

    if st.button("Odeslat dotaz") and q.strip():
        # 1) chybí klíč?
        if st.session_state.openai_client is None:
            st.warning("Nejdřív doplň `OPENAI_API_KEY` do Settings → Secrets.")
        # 2) není postavený index?
        elif st.session_state.index is None:
            st.warning("Nejdřív načti zdroje z assets a postav index (tlačítko vlevo).")
        # 3) všechno OK → vyhledat kontext a zavolat model
        else:
            topk = search_similar(q, k=6)
            ctx_blocks = []
            for d in topk:
                meta = d["meta"]
                ref = f'{meta["source"]}{f" s.{meta["page"]}" if meta["page"] else ""}'
                ctx_blocks.append(f"[{ref}] {d['text'][:800]}")

            prompt = USR_RAG.format(q=q, ctx="\n\n".join(ctx_blocks))

            resp = safe_chat_completion(
                client=st.session_state.openai_client,
                model=MODEL_CHAT,
                messages=[
                    {"role": "system", "content": SYS_RAG},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            st.markdown(resp.choices[0].message.content)

with col2:
    st.subheader("🌦️ Počasí & plán")
    try:
        w = get_weather(city)
        st.metric("Teplota", f"{w['temp']} °C")
        st.caption(f"{w['city']}: {w['desc']} | vítr {w['wind']} km/h")
        st.markdown("[🌦 Zobrazit radar na pocasiaradar.cz](https://www.pocasiaradar.cz/)")
        ctx = weather_context(w)
    except Exception:
        st.warning("Nelze načíst počasí – používám offline hodnoty.")
        w, ctx = {"city": city, "temp": 10, "desc": "offline data", "wind": 0}, "indoor"

    try:
        races = json.loads(races_str)
    except Exception:
        races = []

    pz = periodization(date.today(), None, micro_week, age_group)
if generate_clicked:
    with st.spinner("💪 Generuji plán podle nastavení..."):
        # 1) Načtení počasí
        weather = get_weather(city)
        ctx = "indoor" if weather and weather.get("indoor") else "outdoor"
        
        # 2) Periodizace
        pz = periodization(date.today(), None, micro_week, age_group)
        
        # 3) Vygeneruj plán
        base_plan = generate_plan(age_group, ctx, pz, races, focuses, sessions_per_week)
        
        # 4) Ulož do session (abychom mohli zobrazit později)
        st.session_state["generated_plan"] = base_plan
        
        # 5) Připrav prompt pro čitelnou verzi
        city_desc = (
        f"{weather.get('city','')}: {weather.get('desc','')}"
        + (f" ({weather.get('temp')} °C)" if weather and weather.get('temp') is not None else "")
        )
        
        prompt = USR_PLAN.format(
        base=json.dumps(base_plan, ensure_ascii=False, indent=2),
        age=age_group,
        city_desc=city_desc,
        micro_week=pz["micro_week"],
        deload=pz["deload"],
        )
        
        resp = safe_chat_completion(
        client=st.session_state.openai_client,
        model=MODEL_CHAT,
        messages=[
        {"role": "system", "content": SYS_PLAN},
        {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        )
        
        # 6) Po skončení spinneru – výstup
        st.success("✅ Tréninkový plán byl úspěšně vygenerován!")
        st.markdown(resp.choices[0].message.content)

    st.download_button(
        "📥 Stáhnout plán (JSON)",
        data=json.dumps(base_plan, ensure_ascii=False, indent=2),
        file_name=f"plan_{date.today().isoformat()}.json",
        mime="application/json",
    )











