"""
BDiMO 2026 — Local Evaluation Script

Leave-one-out style evaluation on training data.
For each item, we exclude it from lookups and measure prediction accuracy.

Usage:
    python eval.py                  # run with default settings
    python eval.py --seed 42        # specific random seed
    python eval.py --no-anchors     # disable transductive anchors

WARNING: Local eval underestimates LB by ~1-2% because 666 single-sample
categories have no centroid in LOO. See README for details.
"""
import argparse
import gzip
import os
import pickle
import re
import time
import unicodedata
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'train.tsv')

t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:5.1f}s] {msg}", flush=True)


# ── Text preprocessing (same as main.py) ─────────────────────────────────────

def norm_yo(text):
    return text.replace('ё', 'е').replace('Ё', 'Е') if text else text


BOILERPLATE_WORDS = {
    'доставка', 'возврат', 'оплата', 'гарантия', 'shipping', 'returns',
    'warranty', 'copyright', 'политика', 'условия', 'внимание',
    'корзина', 'добавить', 'купить', 'заказать', 'акция', 'скидка',
    'бесплатно', 'промокод',
}


def clean_text(text):
    if pd.isna(text) or not isinstance(text, str):
        return ''
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&\w+;', ' ', text)
    text = re.sub(r'[«»\u201e\u201c\u201d\u2018\u2019\u2039\u203a\u201b\u201f]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = norm_yo(text.lower())
    return unicodedata.normalize('NFKC', text)


def clean_desc_smart(text):
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
    return re.sub(r'\s+', ' ', ' '.join(kept)).strip()[:500]


def normalize_title(title):
    t = norm_yo(str(title).lower())
    t = re.sub(r'\d+[\.,]?\d*', ' ', t)
    t = re.sub(r'\b(xs|s|m|l|xl|xxl|xxxl)\b', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


SIZE_COLOR_WORDS = {
    'черный', 'белый', 'красный', 'синий', 'зеленый', 'серый', 'розовый',
    'голубой', 'желтый', 'оранжевый', 'фиолетовый', 'коричневый', 'бежевый',
    'черная', 'белая', 'красная', 'синяя', 'серая', 'розовая',
    'черное', 'белое', 'красное', 'синее', 'серое',
    'xs', 's', 'm', 'l', 'xl', 'xxl', 'xxxl',
    'black', 'white', 'red', 'blue', 'green', 'grey', 'gray', 'pink',
}
NOISE_WORDS = {'для', 'с', 'и', 'в', 'на', 'из', 'от', 'по', 'к', 'до', 'не', 'а', 'о', 'у',
               'без', 'что', 'или', 'это', 'его', 'вы', 'мы', 'их',
               'the', 'for', 'and', 'with', 'in', 'of', 'to', 'a', 'an'}
GENERIC_WORDS = {
    'набор', 'комплект', 'портативный', 'портативная', 'портативное',
    'электрический', 'электрическая', 'электрическое',
    'универсальный', 'универсальная', 'универсальное',
    'детский', 'детская', 'детское', 'детские',
    'умный', 'умная', 'умное', 'мини', 'mini',
    'новый', 'новая', 'новое', 'новые',
    'большой', 'большая', 'большое', 'маленький', 'маленькая', 'маленькое',
    'бестселлер', 'премиальный', 'премиальная',
    'эксклюзивный', 'эксклюзивная', 'эксклюзивное',
    'профессиональный', 'профессиональная',
}


def clean_title_aggressive(title):
    t = norm_yo(str(title).lower())
    t = re.sub(r'\d+[\.,]?\d*\s*(см|мм|м|мл|л|г|кг|шт|вт|в|а|штук|пар|x|х|°|%|d|cm|mm|ml|kg|w|v|pcs|pc|m|g)', ' ', t)
    t = re.sub(r'\d+[\.,]?\d*', ' ', t)
    t = re.sub(r'\b(xs|s|m|l|xl|xxl|xxxl)\b', ' ', t)
    for color in SIZE_COLOR_WORDS:
        t = re.sub(r'\b' + re.escape(color) + r'\b', ' ', t)
    t = re.sub(r'[,/\-+\(\)\[\]\"\'«»]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def extract_product_type(title, n=4):
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


def build_text(row):
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
    return extract_product_type(row.get('title', ''), n=6)


# ── Lookups ──────────────────────────────────────────────────────────────────

def build_lookups(df):
    cat_to_dept = df.groupby('category_id')['department_id'].first().to_dict()
    scn_to_cat = {}
    for scn, cats in df.groupby('shop_category_name')['category_id']:
        u = cats.unique()
        if len(u) == 1 and str(scn) != '-' and pd.notna(scn):
            scn_to_cat[norm_yo(str(scn))] = int(u[0])

    combo_to_cat, combo_counts = {}, {}
    for (vn, scn), cats in df.groupby(['vendor_name', 'shop_category_name'])['category_id']:
        u = cats.unique()
        if len(u) == 1 and not is_no_brand(vn):
            k = (norm_yo(str(vn)), norm_yo(str(scn)))
            combo_to_cat[k] = int(u[0])
            combo_counts[k] = len(cats)
    combo_to_cat = {k: v for k, v in combo_to_cat.items() if combo_counts.get(k, 0) >= 2}

    title_scn_to_cat = {}
    for (title, scn), cats in df.groupby(['title', 'shop_category_name'])['category_id']:
        u = cats.unique()
        if len(u) == 1:
            title_scn_to_cat[(norm_yo(str(title)), norm_yo(str(scn)))] = int(u[0])

    vc_to_cat = {}
    for vc, cats in df.dropna(subset=['vendor_code']).groupby('vendor_code')['category_id']:
        u = cats.unique()
        if len(u) == 1:
            vc_to_cat[str(vc)] = int(u[0])

    norm_ts, conflicts = {}, set()
    for (title, scn), cats in df.groupby(['title', 'shop_category_name'])['category_id']:
        key = (normalize_title(title), norm_yo(str(scn)))
        u = cats.unique()
        if len(u) == 1:
            val = int(u[0])
            if key not in norm_ts:
                norm_ts[key] = val
            elif norm_ts[key] != val:
                conflicts.add(key)
    for k in conflicts:
        del norm_ts[k]

    pt_scn, pt_conflicts = {}, set()
    for _, row in df.iterrows():
        key = (extract_product_type(row['title'], n=3), norm_yo(str(row['shop_category_name'])))
        cat = int(row['category_id'])
        if key not in pt_scn:
            pt_scn[key] = cat
        elif pt_scn[key] != cat:
            pt_conflicts.add(key)
    for k in pt_conflicts:
        pt_scn.pop(k, None)

    title_to_cat = {}
    for title, cats in df.groupby('title')['category_id']:
        u = cats.unique()
        if len(u) == 1:
            title_to_cat[norm_yo(str(title))] = int(u[0])

    return {
        'cat_to_dept': cat_to_dept, 'scn_to_cat': scn_to_cat, 'combo_to_cat': combo_to_cat,
        'title_scn_to_cat': title_scn_to_cat, 'vc_to_cat': vc_to_cat,
        'norm_title_scn_to_cat': norm_ts, 'pt_scn_to_cat': pt_scn, 'title_to_cat': title_to_cat,
    }


def predict_lookups(df_test, lookups):
    n = len(df_test)
    pred = np.full(n, -1, dtype=np.int64)
    source = [''] * n
    combo, title_scn = lookups['combo_to_cat'], lookups['title_scn_to_cat']
    norm_ts, scn, vc = lookups['norm_title_scn_to_cat'], lookups['scn_to_cat'], lookups['vc_to_cat']
    pt_scn, title_alone = lookups['pt_scn_to_cat'], lookups['title_to_cat']

    for i in range(n):
        row = df_test.iloc[i]
        r_vn, r_scn = norm_yo(str(row.get('vendor_name', ''))), norm_yo(str(row.get('shop_category_name', '')))
        r_title, r_vc = norm_yo(str(row.get('title', ''))), str(row.get('vendor_code', ''))

        if (r_vn, r_scn) in combo:
            pred[i], source[i] = combo[(r_vn, r_scn)], 'combo'
        elif (r_title, r_scn) in title_scn:
            pred[i], source[i] = title_scn[(r_title, r_scn)], 'title_scn'
        elif (normalize_title(r_title), r_scn) in norm_ts:
            pred[i], source[i] = norm_ts[(normalize_title(r_title), r_scn)], 'norm_ts'
        elif r_vc != 'nan' and r_vc in vc:
            pred[i], source[i] = vc[r_vc], 'vc'
        elif (extract_product_type(r_title, n=3), r_scn) in pt_scn:
            pred[i], source[i] = pt_scn[(extract_product_type(r_title, n=3), r_scn)], 'pt_scn'
        elif r_scn in scn:
            pred[i], source[i] = scn[r_scn], 'scn'
        elif r_title in title_alone:
            pred[i], source[i] = title_alone[r_title], 'title'
    return pred, source


# ── Main eval ────────────────────────────────────────────────────────────────

def run_eval(seed=42, use_anchors=True):
    log("Loading data...")
    df = pd.read_csv(TRAIN_PATH, sep='\t')
    log(f"{len(df)} rows, {df.category_id.nunique()} categories")

    # Split: use cats with 2+ samples for eval (leave one out per cat)
    cat_counts = df.groupby('category_id').size()
    eval_cats = set(cat_counts[cat_counts >= 2].index)

    np.random.seed(seed)
    val_idx = []
    train_idx = []
    for cat, group in df.groupby('category_id'):
        idxs = group.index.tolist()
        if cat in eval_cats:
            chosen = np.random.choice(idxs)
            val_idx.append(chosen)
            train_idx.extend([i for i in idxs if i != chosen])
        else:
            train_idx.extend(idxs)

    df_train = df.loc[train_idx].reset_index(drop=True)
    df_val = df.loc[val_idx].reset_index(drop=True)
    log(f"Train: {len(df_train)}, Val: {len(df_val)} ({len(eval_cats)} cats with 2+ samples)")

    log("Building lookups from train split...")
    lookups = build_lookups(df_train)
    cat_to_dept = lookups['cat_to_dept']

    log("Lookup predictions on val...")
    pred_cats, source = predict_lookups(df_val, lookups)
    n_found = (pred_cats != -1).sum()
    log(f"  {n_found}/{len(df_val)} resolved ({n_found/len(df_val)*100:.1f}%)")

    log("Building TF-IDF...")
    train_texts = df_train.apply(build_text, axis=1).tolist()
    train_titles = df_train.apply(build_title_text, axis=1).tolist()
    val_texts = df_val.apply(build_text, axis=1).tolist()
    val_titles = df_val.apply(build_title_text, axis=1).tolist()

    # Semi-supervised: fit on train+val
    all_texts = train_texts + val_texts
    all_titles = train_titles + val_titles
    n_train = len(train_texts)

    tfidf_w = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=80000, sublinear_tf=True, dtype=np.float32)
    tfidf_c = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=80000, sublinear_tf=True, dtype=np.float32)
    tfidf_t = TfidfVectorizer(analyzer='word', ngram_range=(1,2), max_features=30000, sublinear_tf=True, dtype=np.float32)

    Xw = tfidf_w.fit_transform(all_texts)
    Xc = tfidf_c.fit_transform(all_texts)
    Xt = tfidf_t.fit_transform(all_titles)

    Xw_tr, Xc_tr, Xt_tr = Xw[:n_train], Xc[:n_train], Xt[:n_train]
    Xw_val, Xc_val, Xt_val = Xw[n_train:], Xc[n_train:], Xt[n_train:]

    # Transductive anchors
    knn_labels = df_train['category_id'].values
    knn_depts = df_train['department_id'].values
    knn_w, knn_c, knn_t = Xw_tr, Xc_tr, Xt_tr

    if use_anchors:
        HIGH_PURITY = {'combo', 'title_scn', 'norm_ts', 'vc', 'pt_scn'}
        anchor_mask = np.array([pred_cats[i] != -1 and source[i] in HIGH_PURITY for i in range(len(df_val))])
        n_anc = anchor_mask.sum()
        if n_anc > 0:
            anc_labels = pred_cats[anchor_mask]
            anc_depts = np.array([cat_to_dept.get(int(c), 9) for c in anc_labels])
            knn_labels = np.concatenate([knn_labels, anc_labels])
            knn_depts = np.concatenate([knn_depts, anc_depts])
            knn_w = sparse.vstack([knn_w, Xw_val[anchor_mask]])
            knn_c = sparse.vstack([knn_c, Xc_val[anchor_mask]])
            knn_t = sparse.vstack([knn_t, Xt_val[anchor_mask]])
            log(f"  Anchors: {n_anc} items added to index")

    # Centroids
    log("Building centroids...")
    unique_cats = np.unique(knn_labels)
    cat_idx = {c: i for i, c in enumerate(unique_cats)}
    n_cats = len(unique_cats)

    row_ind = [cat_idx[c] for c in knn_labels]
    col_ind = list(range(len(knn_labels)))
    indicator = sparse.csr_matrix((np.ones(len(knn_labels), dtype=np.float32), (row_ind, col_ind)), shape=(n_cats, len(knn_labels)))
    counts = np.array(indicator.sum(axis=1)).flatten()
    counts[counts == 0] = 1
    indicator = sparse.diags(1.0 / counts) @ indicator

    Cw = normalize(indicator @ knn_w)
    Cc = normalize(indicator @ knn_c)
    Ct = normalize(indicator @ knn_t)

    # Dept model
    X_comb = sparse.hstack([Xw_tr, Xc_tr, Xt_tr], format='csr')
    dept_model = LinearSVC(C=1.0, max_iter=2000, random_state=seed)
    dept_model.fit(X_comb, df_train['department_id'].values)

    dept_to_cats = {}
    for d, cats in df_train.groupby('department_id')['category_id']:
        dept_to_cats[int(d)] = set(cats.unique())
    dept_to_cidx = {d: set(cat_idx[c] for c in cs if c in cat_idx) for d, cs in dept_to_cats.items()}

    X_val_comb = sparse.hstack([Xw_val, Xc_val, Xt_val], format='csr')
    pred_depts = dept_model.predict(X_val_comb)

    # kNN scoring
    log("kNN scoring...")
    n = len(df_val)
    knn_pred = np.zeros(n, dtype=np.int64)

    for start in range(0, n, 500):
        end = min(start + 500, n)
        sl = slice(start, end)
        sim = 0.10 * cosine_similarity(Xw_val[sl], Cw) + 0.70 * cosine_similarity(Xc_val[sl], Cc) + 0.20 * cosine_similarity(Xt_val[sl], Ct)
        for i in range(end - start):
            gi = start + i
            in_dept = dept_to_cidx.get(int(pred_depts[gi]), set())
            s = sim[i].copy()
            for j in range(len(s)):
                if j not in in_dept:
                    s[j] *= 0.6
            knn_pred[gi] = unique_cats[np.argmax(s)]

    # Merge
    need = np.where(pred_cats == -1)[0]
    pred_cats[need] = knn_pred[need]

    true_cats = df_val['category_id'].values
    true_depts = df_val['department_id'].values
    pred_depts_final = np.array([cat_to_dept.get(int(c), 9) for c in pred_cats])

    cat_acc = (pred_cats == true_cats).mean()
    dept_acc = (pred_depts_final == true_depts).mean()
    combined = (cat_acc + dept_acc) / 2

    log(f"\n{'='*50}")
    log(f"  cat_acc:  {cat_acc*100:.2f}%")
    log(f"  dept_acc: {dept_acc*100:.2f}%")
    log(f"  combined: {combined*100:.2f}%")
    log(f"{'='*50}")

    # Breakdown
    lookup_mask = np.array([s != '' for s in source])
    knn_mask = ~lookup_mask
    if lookup_mask.sum() > 0:
        log(f"  Lookup items ({lookup_mask.sum()}): cat_acc={((pred_cats[lookup_mask]==true_cats[lookup_mask]).mean()*100):.2f}%")
    if knn_mask.sum() > 0:
        log(f"  kNN items ({knn_mask.sum()}):    cat_acc={((pred_cats[knn_mask]==true_cats[knn_mask]).mean()*100):.2f}%")

    log(f"\nDone in {time.time()-t0:.0f}s")
    return combined


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BDiMO 2026 - Local Evaluation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for LOO split')
    parser.add_argument('--no-anchors', action='store_true', help='Disable transductive anchors')
    args = parser.parse_args()

    run_eval(seed=args.seed, use_anchors=not args.no_anchors)
