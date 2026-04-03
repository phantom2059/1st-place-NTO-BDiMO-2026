"""
BDiMO 2026 - 1st Place Solution (Inference)
Public LB: 76.36%

Pipeline:
  1. Lookup cascade — resolve ~40% of items via exact-match dictionaries
  2. Semi-supervised TF-IDF — fit vectorizers on train+test (no label leak)
  3. Transductive anchors — feed confident lookup results back into kNN index
  4. Centroid kNN — 3-channel weighted cosine similarity against category centroids
  5. Department derivation — category_id -> department_id (1:1 mapping)
"""
import gzip
import os
import pickle
import re
import time
import unicodedata

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_PATH = os.path.join(BASE_DIR, 'test.tsv')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
OUTPUT_PATH = os.path.join(BASE_DIR, 'prediction.csv')

t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


# ── Text Preprocessing ───────────────────────────────────────────────────────

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
    text = re.sub(r'[«»\u201e\u201c\u201d\u2018\u2019\u2039\u203a\u201b\u201f]', '', text)
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
    """Take first N meaningful words from title — captures product type.
    "Новый Электрический Чайник Bosch 1.7л белый" -> "чайник bosch"
    """
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
    """Build text for word/char TF-IDF: product_type*3 + title + SCN*3 + desc + vendor.
    Repetitions boost TF-IDF weight of the most discriminative fields.
    """
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


# ── Artifact Loading ─────────────────────────────────────────────────────────

def load_artifact(name):
    path = os.path.join(ARTIFACTS_DIR, name)
    if name.endswith('.pkl.gz'):
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)
    elif name.endswith('.npz'):
        return sparse.load_npz(path)


# ── Stage 1: Lookup Cascade ──────────────────────────────────────────────────

def predict_lookups(df_test, lookups):
    """Try to resolve each item via exact-match lookup tables.
    Priority order (highest confidence first):
      combo > title_scn > norm_ts > vc > pt_scn > scn > title
    First match wins — no need for ML on these items.
    """
    n = len(df_test)
    pred = np.full(n, -1, dtype=np.int64)
    source = [''] * n
    combo = lookups['combo_to_cat']
    title_scn = lookups['title_scn_to_cat']
    norm_title_scn = lookups.get('norm_title_scn_to_cat', {})
    scn = lookups['scn_to_cat']
    vc = lookups['vc_to_cat']
    pt_scn = lookups.get('pt_scn_to_cat', {})
    title_alone = lookups.get('title_to_cat', {})

    for i in range(n):
        row = df_test.iloc[i]
        r_vn = norm_yo(str(row.get('vendor_name', '')))
        r_scn = norm_yo(str(row.get('shop_category_name', '')))
        r_title = norm_yo(str(row.get('title', '')))
        r_vc = str(row.get('vendor_code', ''))

        if (r_vn, r_scn) in combo:
            pred[i], source[i] = combo[(r_vn, r_scn)], 'combo'
        elif (r_title, r_scn) in title_scn:
            pred[i], source[i] = title_scn[(r_title, r_scn)], 'title_scn'
        elif (normalize_title(r_title), r_scn) in norm_title_scn:
            pred[i], source[i] = norm_title_scn[(normalize_title(r_title), r_scn)], 'norm_ts'
        elif r_vc != 'nan' and r_vc in vc:
            pred[i], source[i] = vc[r_vc], 'vc'
        elif (extract_product_type(r_title, n=3), r_scn) in pt_scn:
            pred[i], source[i] = pt_scn[(extract_product_type(r_title, n=3), r_scn)], 'pt_scn'
        elif r_scn in scn:
            pred[i], source[i] = scn[r_scn], 'scn'
        elif r_title in title_alone:
            pred[i], source[i] = title_alone[r_title], 'title'

    return pred, source


# ── Main Pipeline ────────────────────────────────────────────────────────────

def main():
    log("Loading test data...")
    df_test = pd.read_csv(TEST_PATH, sep='\t')
    log(f"{len(df_test)} samples")

    log("Loading artifacts...")
    lookups = load_artifact('lookups.pkl.gz')
    train_labels = load_artifact('train_labels.pkl.gz')
    train_depts = load_artifact('train_depts.pkl.gz')
    dept_to_cats = load_artifact('dept_to_cats.pkl.gz')
    train_texts = load_artifact('train_texts.pkl.gz')
    train_titles = load_artifact('train_titles.pkl.gz')
    cat_to_dept = lookups['cat_to_dept']
    log("Loaded")

    # Stage 1: Lookup cascade
    log("Lookups...")
    pred_cats, source = predict_lookups(df_test, lookups)
    n_found = (pred_cats != -1).sum()
    log(f"  {n_found}/{len(df_test)} resolved ({n_found/len(df_test)*100:.1f}%)")

    # Semi-supervised TF-IDF: fit on train+test combined (vocabulary expansion)
    log("Building semi-supervised TF-IDF...")
    test_texts = df_test.apply(build_combined_text, axis=1).values
    test_titles = df_test.apply(build_title_text, axis=1).values

    all_texts = list(train_texts) + list(test_texts)
    all_titles = list(train_titles) + list(test_titles)
    n_train = len(train_texts)

    tfidf_word = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 2), max_features=80000,
        sublinear_tf=True, min_df=1, max_df=0.95, dtype=np.float32,
    )
    # Char n-grams: most important channel (weight 0.70)
    # Robust to Russian morphology — no lemmatizer needed
    tfidf_char = TfidfVectorizer(
        analyzer='char_wb', ngram_range=(3, 5), max_features=80000,
        sublinear_tf=True, min_df=1, max_df=0.95, dtype=np.float32,
    )
    tfidf_title = TfidfVectorizer(
        analyzer='word', ngram_range=(1, 2), max_features=30000,
        sublinear_tf=True, min_df=1, max_df=0.95, dtype=np.float32,
    )

    X_all_word = tfidf_word.fit_transform(all_texts)
    X_all_char = tfidf_char.fit_transform(all_texts)
    X_all_title = tfidf_title.fit_transform(all_titles)
    log(f"  word={X_all_word.shape}, char={X_all_char.shape}, title={X_all_title.shape}")

    X_train_word = X_all_word[:n_train]
    X_train_char = X_all_char[:n_train]
    X_train_title = X_all_title[:n_train]
    X_test_word = X_all_word[n_train:]
    X_test_char = X_all_char[n_train:]
    X_test_title = X_all_title[n_train:]

    # Transductive anchor expansion: confident lookup results -> kNN index
    # Adds ~2000 near-perfectly labeled items, enriching rare-category centroids
    HIGH_PURITY_SOURCES = {'combo', 'title_scn', 'norm_ts', 'vc', 'pt_scn'}
    anchor_mask = np.array([
        pred_cats[i] != -1 and source[i] in HIGH_PURITY_SOURCES
        for i in range(len(df_test))
    ])
    n_anchors = anchor_mask.sum()
    log(f"  Transductive anchors: {n_anchors} items from {HIGH_PURITY_SOURCES}")

    if n_anchors > 0:
        anchor_labels = pred_cats[anchor_mask].copy()
        anchor_depts = np.array([cat_to_dept.get(int(c), 9) for c in anchor_labels])

        knn_labels = np.concatenate([train_labels, anchor_labels])
        knn_depts = np.concatenate([train_depts, anchor_depts])
        knn_word = sparse.vstack([X_train_word, X_test_word[anchor_mask]])
        knn_char = sparse.vstack([X_train_char, X_test_char[anchor_mask]])
        knn_title = sparse.vstack([X_train_title, X_test_title[anchor_mask]])
        log(f"  Expanded KNN index: {len(knn_labels)} items (was {n_train})")
    else:
        knn_labels = train_labels
        knn_depts = train_depts
        knn_word, knn_char, knn_title = X_train_word, X_train_char, X_train_title

    # Stage 2: Centroid-based kNN
    # One centroid per category = average TF-IDF vector, L2-normalized
    # With 1-3 samples per category, centroids smooth out noise
    log("Computing category centroids...")
    unique_cats = np.unique(knn_labels)
    cat_idx_map = {c: i for i, c in enumerate(unique_cats)}
    n_cats = len(unique_cats)

    # Sparse indicator matrix: centroids = normalize(indicator @ X)
    row_ind = [cat_idx_map[c] for c in knn_labels]
    col_ind = list(range(len(knn_labels)))
    ind_data = np.ones(len(knn_labels), dtype=np.float32)
    indicator = sparse.csr_matrix((ind_data, (row_ind, col_ind)), shape=(n_cats, len(knn_labels)))
    counts = np.array(indicator.sum(axis=1)).flatten()
    counts[counts == 0] = 1
    indicator = sparse.diags(1.0 / counts) @ indicator

    C_word = normalize(indicator @ knn_word, norm='l2')
    C_char = normalize(indicator @ knn_char, norm='l2')
    C_title = normalize(indicator @ knn_title, norm='l2')
    log(f"  {n_cats} centroids computed")

    # Department model for soft constraint
    log("Training dept model...")
    X_train_comb = sparse.hstack([X_train_word, X_train_char, X_train_title], format='csr')
    dept_model = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    dept_model.fit(X_train_comb, train_depts)
    log("  Done")

    # kNN scoring: weighted 3-channel cosine + department penalty
    log("kNN (centroid-based)...")
    n = len(df_test)
    knn_pred = np.zeros(n, dtype=np.int64)
    knn_sim_arr = np.zeros(n, dtype=np.float32)

    X_wct_test = sparse.hstack([X_test_word, X_test_char, X_test_title], format='csr')
    pred_depts = dept_model.predict(X_wct_test)
    dept_to_cidx = {}
    for d, cats in dept_to_cats.items():
        dept_to_cidx[int(d)] = set(cat_idx_map[c] for c in cats if c in cat_idx_map)

    batch = 500
    for start in range(0, n, batch):
        end = min(start + batch, n)
        sl = slice(start, end)

        sim_w = cosine_similarity(X_test_word[sl], C_word)
        sim_c = cosine_similarity(X_test_char[sl], C_char)
        sim_t = cosine_similarity(X_test_title[sl], C_title)
        cent_s = 0.10 * sim_w + 0.70 * sim_c + 0.20 * sim_t

        for i in range(end - start):
            gi = start + i
            in_dept = dept_to_cidx.get(int(pred_depts[gi]), set())
            sim = cent_s[i].copy()
            # Soft dept constraint: 0.6x penalty for out-of-department categories
            for j in range(len(sim)):
                if j not in in_dept:
                    sim[j] *= 0.6
            best = np.argmax(sim)
            knn_pred[gi] = unique_cats[best]
            knn_sim_arr[gi] = sim[best]

        if end % 1000 == 0 or end == n:
            log(f"    {end}/{n}")

    # Merge: lookups + kNN
    need = np.where(pred_cats == -1)[0]
    pred_cats[need] = knn_pred[need]

    # Override low-confidence lookups when kNN strongly disagrees
    overrides = 0
    for i in range(n):
        if source[i] == 'combo' and knn_sim_arr[i] >= 0.6 and knn_pred[i] != pred_cats[i]:
            pred_cats[i] = knn_pred[i]
            overrides += 1
        elif source[i] == 'title' and knn_sim_arr[i] >= 0.3 and knn_pred[i] != pred_cats[i]:
            pred_cats[i] = knn_pred[i]
            overrides += 1
    if overrides > 0:
        log(f"  Overrides: {overrides}")

    # category_id -> department_id (1:1 mapping)
    pred_depts_final = np.array([cat_to_dept.get(int(c), 9) for c in pred_cats])

    result = pd.DataFrame({
        'category_id': pred_cats,
        'department_id': pred_depts_final,
    })
    result.to_csv(OUTPUT_PATH, index=False)
    log(f"Saved {len(result)} predictions, total {time.time()-t0:.1f}s")


main()
