"""
BDiMO 2026 — 1st Place Solution (Training / Artifact Building)

Builds lookup tables and precomputed training texts from train.tsv.
Artifacts are saved to artifacts/ (~3.3 MB total).
Run time: ~4 seconds.
"""
import gzip
import os
import pickle
import re
import time
import unicodedata
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'train.tsv')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')

t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


# ── Text Preprocessing (shared with main.py) ─────────────────────────────────

def norm_yo(text):
    """ё and е are used inconsistently in Russian product listings — normalize."""
    return text.replace('ё', 'е').replace('Ё', 'Е') if text else text


BOILERPLATE_WORDS = {
    'доставка', 'возврат', 'оплата', 'гарантия', 'shipping', 'returns',
    'warranty', 'copyright', 'политика', 'условия', 'внимание',
    'корзина', 'добавить', 'купить', 'заказать', 'акция', 'скидка',
    'бесплатно', 'промокод',
}


def clean_text(text):
    """Strip HTML, quotes, normalize unicode."""
    if pd.isna(text) or not isinstance(text, str):
        return ''
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&\w+;', ' ', text)
    text = re.sub(r'[\u00ab\u00bb\u201e\u201c\u201d\u2018\u2019\u2039\u203a\u201b\u201f]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = norm_yo(text.lower())
    return unicodedata.normalize('NFKC', text)


def clean_desc_smart(text):
    """Split description into blocks, drop boilerplate (shipping, returns, URLs)."""
    if pd.isna(text) or not isinstance(text, str):
        return ''
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&\w+;', ' ', text)
    blocks = re.split(r'\n{2,}|<br\s*/?>|</p>|</div>', text)
    kept = []
    for block in blocks:
        block = block.strip()
        if len(block) < 10:
            continue
        block_lower = block.lower()
        if sum(1 for w in BOILERPLATE_WORDS if w in block_lower) >= 2:
            continue
        if block.count('http') + block.count('www') + block.count('.ru/') >= 2:
            continue
        kept.append(block)
    result = ' '.join(kept)
    return re.sub(r'\s+', ' ', result).strip()[:500]


def normalize_title(title):
    """Light normalization: strip numbers and sizes for fuzzy lookup matching."""
    t = norm_yo(str(title).lower())
    t = re.sub(r'\d+[\.,]?\d*', ' ', t)
    t = re.sub(r'\b(xs|s|m|l|xl|xxl|xxxl)\b', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


SIZE_COLOR_WORDS = {
    'черный', 'белый', 'красный', 'синий', 'зеленый', 'серый', 'розовый',
    'голубой', 'желтый', 'оранжевый', 'фиолетовый', 'коричневый', 'бежевый',
    'черная', 'белая', 'красная', 'синяя', 'серая', 'розовая',
    'черное', 'белое', 'красное', 'синее', 'серое',
    'xs', 's', 'm', 'l', 'xl', 'xxl', 'xxxl',
    'black', 'white', 'red', 'blue', 'green', 'grey', 'gray', 'pink',
}

NOISE_WORDS = {
    'для', 'с', 'и', 'в', 'на', 'из', 'от', 'по', 'к', 'до', 'не', 'а', 'о', 'у',
    'без', 'что', 'или', 'это', 'его', 'вы', 'мы', 'их',
    'the', 'for', 'and', 'with', 'in', 'of', 'to', 'a', 'an',
}

GENERIC_WORDS = {
    'набор', 'комплект', 'портативный', 'портативная', 'портативное',
    'электрический', 'электрическая', 'электрическое',
    'универсальный', 'универсальная', 'универсальное',
    'детский', 'детская', 'детское', 'детские',
    'умный', 'умная', 'умное',
    'эксклюзивный', 'эксклюзивная', 'эксклюзивное',
    'профессиональный', 'профессиональная',
    'мини', 'mini', 'новый', 'новая', 'новое', 'новые',
    'большой', 'большая', 'большое',
    'маленький', 'маленькая', 'маленькое',
    'бестселлер', 'премиальный', 'премиальная',
}


def clean_title_aggressive(title):
    """Strip numbers+units, sizes, colors, punctuation from title."""
    t = norm_yo(str(title).lower())
    t = re.sub(r'\d+[\.,]?\d*\s*(см|мм|м|мл|л|г|кг|шт|вт|в|а|штук|пар|x|х|°|%|d|cm|mm|ml|kg|w|v|pcs|pc|m|g)', ' ', t)
    t = re.sub(r'\d+[\.,]?\d*', ' ', t)
    t = re.sub(r'\b(xs|s|m|l|xl|xxl|xxxl)\b', ' ', t)
    for color in SIZE_COLOR_WORDS:
        t = re.sub(r'\b' + re.escape(color) + r'\b', ' ', t)
    t = re.sub(r'[,/\-+\(\)\[\]\"\'«»]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def extract_product_type(title, n=4):
    """Take first N meaningful words from title — captures product type."""
    t = clean_title_aggressive(title)
    words = t.split()
    meaningful = []
    for w in words:
        if w not in NOISE_WORDS and w not in GENERIC_WORDS and len(w) > 1:
            meaningful.append(w)
            if len(meaningful) >= n:
                break
    return ' '.join(meaningful)


def is_no_brand(v):
    return str(v).lower().strip() in ('нет бренда', 'без бренда', 'no brand', '没有品牌', 'nan', '')


def build_combined_text(row):
    """Build text for word/char TF-IDF: product_type*3 + title + SCN*3 + desc + vendor."""
    title = clean_text(row.get('title', ''))
    product = extract_product_type(row.get('title', ''), n=4)
    shop_cat = clean_text(row.get('shop_category_name', ''))
    desc = clean_desc_smart(row.get('description', ''))
    vendor = clean_text(row.get('vendor_name', ''))
    parts = [product, product, product, title, shop_cat, shop_cat, shop_cat]
    if desc:
        parts.append(desc)
    if vendor and not is_no_brand(vendor):
        parts.append(vendor)
    return ' '.join(p for p in parts if p)


def build_title_text(row):
    """Build text for title-only TF-IDF channel (product type, 6 words)."""
    return extract_product_type(row.get('title', ''), n=6)


# ── Lookup Table Construction ─────────────────────────────────────────────────

def build_lookups(df):
    """Build 7-level lookup cascade from training data.

    Each lookup maps a combination of fields -> category_id,
    but only when the mapping is unambiguous (all samples agree).
    """
    cat_to_dept = df.groupby('category_id')['department_id'].first().to_dict()

    # Level 6: shop_category_name -> category (42% of SCNs are unique)
    scn_to_cat = {}
    for scn, cats in df.groupby('shop_category_name')['category_id']:
        u = cats.unique()
        if len(u) == 1 and str(scn) != '-' and pd.notna(scn):
            scn_to_cat[norm_yo(str(scn))] = int(u[0])

    # Level 1: (vendor_name, shop_category_name) -> category, min 2 samples
    combo_to_cat = {}
    combo_counts = {}
    for (vn, scn), cats in df.groupby(['vendor_name', 'shop_category_name'])['category_id']:
        u = cats.unique()
        if len(u) == 1 and not is_no_brand(vn):
            combo_to_cat[(norm_yo(str(vn)), norm_yo(str(scn)))] = int(u[0])
            combo_counts[(norm_yo(str(vn)), norm_yo(str(scn)))] = len(cats)
    combo_to_cat = {k: v for k, v in combo_to_cat.items() if combo_counts.get(k, 0) >= 2}

    # Level 2: (title, shop_category_name) -> category
    title_scn_to_cat = {}
    for (title, scn), cats in df.groupby(['title', 'shop_category_name'])['category_id']:
        u = cats.unique()
        if len(u) == 1:
            title_scn_to_cat[(norm_yo(str(title)), norm_yo(str(scn)))] = int(u[0])

    # Level 4: vendor_code -> category
    vc_to_cat = {}
    df_vc = df.dropna(subset=['vendor_code'])
    for vc, cats in df_vc.groupby('vendor_code')['category_id']:
        u = cats.unique()
        if len(u) == 1:
            vc_to_cat[str(vc)] = int(u[0])

    # Level 3: (normalized_title, shop_category_name) -> category
    norm_title_scn_to_cat = {}
    conflicts = set()
    for (title, scn), cats in df.groupby(['title', 'shop_category_name'])['category_id']:
        norm_t = normalize_title(title)
        key = (norm_t, norm_yo(str(scn)))
        u = cats.unique()
        if len(u) == 1:
            val = int(u[0])
            if key not in norm_title_scn_to_cat:
                norm_title_scn_to_cat[key] = val
            elif norm_title_scn_to_cat[key] != val:
                conflicts.add(key)
    for k in conflicts:
        del norm_title_scn_to_cat[k]

    # Level 5: (product_type, shop_category_name) -> category
    pt_scn_to_cat = {}
    pt_scn_conflicts = set()
    for i, row in df.iterrows():
        pt = extract_product_type(row['title'], n=3)
        scn = norm_yo(str(row['shop_category_name']))
        key = (pt, scn)
        cat = int(row['category_id'])
        if key not in pt_scn_to_cat:
            pt_scn_to_cat[key] = cat
        elif pt_scn_to_cat[key] != cat:
            pt_scn_conflicts.add(key)
    for k in pt_scn_conflicts:
        if k in pt_scn_to_cat:
            del pt_scn_to_cat[k]

    # Level 7: title -> category (lowest priority)
    title_to_cat = {}
    for title, cats in df.groupby('title')['category_id']:
        u = cats.unique()
        if len(u) == 1:
            title_to_cat[norm_yo(str(title))] = int(u[0])

    return {
        'cat_to_dept': cat_to_dept,
        'scn_to_cat': scn_to_cat,
        'combo_to_cat': combo_to_cat,
        'title_scn_to_cat': title_scn_to_cat,
        'vc_to_cat': vc_to_cat,
        'norm_title_scn_to_cat': norm_title_scn_to_cat,
        'pt_scn_to_cat': pt_scn_to_cat,
        'title_to_cat': title_to_cat,
    }


# ── Artifact Saving ──────────────────────────────────────────────────────────

def save_pkl(obj, name):
    with gzip.open(os.path.join(ARTIFACTS_DIR, name), 'wb') as f:
        pickle.dump(obj, f)


def main():
    log("Loading data...")
    df = pd.read_csv(TRAIN_PATH, sep='\t')
    log(f"{len(df)} rows, {df['category_id'].nunique()} categories")

    log("Building lookups...")
    lookups = build_lookups(df)
    for k, v in lookups.items():
        log(f"  {k}: {len(v)}")

    log("Building texts for semi-supervised TF-IDF...")
    texts_combined = df.apply(build_combined_text, axis=1).values
    texts_title = df.apply(build_title_text, axis=1).values

    dept_to_cats = {}
    for dept, cats in df.groupby('department_id')['category_id']:
        dept_to_cats[int(dept)] = list(set(cats.values.tolist()))

    log("Saving artifacts...")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    save_pkl(lookups, 'lookups.pkl.gz')
    save_pkl(df['category_id'].values, 'train_labels.pkl.gz')
    save_pkl(df['department_id'].values, 'train_depts.pkl.gz')
    save_pkl(dept_to_cats, 'dept_to_cats.pkl.gz')
    save_pkl(list(texts_combined), 'train_texts.pkl.gz')
    save_pkl(list(texts_title), 'train_titles.pkl.gz')

    total_size = 0
    for fname in sorted(os.listdir(ARTIFACTS_DIR)):
        fpath = os.path.join(ARTIFACTS_DIR, fname)
        size = os.path.getsize(fpath)
        total_size += size
        log(f"  {fname}: {size/1024/1024:.1f} MB")
    log(f"  Total: {total_size/1024/1024:.1f} MB")
    log(f"Done in {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
