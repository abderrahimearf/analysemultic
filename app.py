import streamlit as st
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(page_title="arfaoui AMC", layout="wide")

# TABLEAU RI STANDARD (COURS)
RI_DICT_STD = {
    1: 0.0, 2: 0.0, 
    3: 0.58, 4: 0.90, 5: 1.12, 
    6: 1.24, 7: 1.32, 8: 1.41, 
    9: 1.45, 10: 1.56
}

# ==========================================
# 1. FONCTIONS MATHÉMATIQUES (CORE)
# ==========================================

def calculate_entropy_weights(df):
    matrix = df.values.astype(float)
    m, n = matrix.shape
    col_sums = matrix.sum(axis=0)
    p_ij = np.divide(matrix, col_sums, where=col_sums!=0)
    k = 1 / np.log(m) if m > 1 else 1
    epsilon = 1e-12
    p_ij_log = np.log(p_ij + epsilon)
    e_j = -k * np.sum(p_ij * p_ij_log, axis=0)
    d_j = 1 - e_j
    sum_d = d_j.sum()
    if sum_d == 0: return np.ones(n) / n
    return d_j / sum_d

def calculate_critic_weights(df, criteria_types):
    matrix = df.values.astype(float)
    m, n = matrix.shape
    norm_matrix = np.zeros_like(matrix)
    for j in range(n):
        col = matrix[:, j]
        min_val, max_val = np.min(col), np.max(col)
        denom = max_val - min_val
        if denom == 0: norm_matrix[:, j] = 0
        else:
            if criteria_types[j] == 'Bénéfice (+)':
                norm_matrix[:, j] = (col - min_val) / denom
            else:
                norm_matrix[:, j] = (max_val - col) / denom
    std_dev = np.std(norm_matrix, axis=0, ddof=1)
    df_norm = pd.DataFrame(norm_matrix)
    corr_matrix = df_norm.corr().fillna(0).values
    c_j = std_dev * np.sum(1 - corr_matrix, axis=0)
    sum_cj = np.sum(c_j)
    if sum_cj == 0: return np.ones(n) / n
    return np.nan_to_num(c_j / sum_cj, nan=0.0)

def run_waspas(df, weights, criteria_types, lmbda=0.5):
    matrix = df.values.astype(float)
    norm_matrix = np.zeros_like(matrix)
    m, n = matrix.shape
    for j in range(n):
        col = matrix[:, j]
        max_val = np.max(col)
        min_val = np.min(col)
        if criteria_types[j] == 'Bénéfice (+)':
            norm_matrix[:, j] = np.divide(col, max_val, where=max_val!=0)
        else:
            norm_matrix[:, j] = np.divide(min_val, col, where=col!=0)
    wsm = np.sum(norm_matrix * weights, axis=1)
    weighted_power_matrix = np.power(norm_matrix, weights)
    wpm = np.prod(weighted_power_matrix, axis=1)
    qi = (lmbda * wsm) + ((1 - lmbda) * wpm)
    return wsm, wpm, qi

def run_topsis(df, weights, criteria_types):
    matrix = df.values.astype(float)
    denominators = np.sqrt(np.sum(matrix**2, axis=0))
    norm_matrix = np.divide(matrix, denominators, where=denominators!=0)
    weighted_matrix = norm_matrix * np.nan_to_num(weights, nan=0.0)
    ideal_best, ideal_worst = [], []
    for j in range(len(criteria_types)):
        col = weighted_matrix[:, j]
        if criteria_types[j] == 'Bénéfice (+)':
            ideal_best.append(np.max(col))
            ideal_worst.append(np.min(col))
        else:
            ideal_best.append(np.min(col))
            ideal_worst.append(np.max(col))
    s_plus = np.sqrt(np.sum((weighted_matrix - np.array(ideal_best))**2, axis=1))
    s_minus = np.sqrt(np.sum((weighted_matrix - np.array(ideal_worst))**2, axis=1))
    denom = s_plus + s_minus
    score = np.divide(s_minus, denom, where=denom!=0)
    return score

def run_vikor(df, weights, criteria_types, v=0.5):
    matrix = df.values.astype(float)
    m, n = matrix.shape
    f_star, f_minus = [], []
    for j in range(n):
        col = matrix[:, j]
        if criteria_types[j] == 'Bénéfice (+)':
            f_star.append(np.max(col))
            f_minus.append(np.min(col))
        else:
            f_star.append(np.min(col))
            f_minus.append(np.max(col))
    f_star, f_minus = np.array(f_star), np.array(f_minus)
    s_i, r_i = np.zeros(m), np.zeros(m)
    for i in range(m):
        dists = []
        for j in range(n):
            denom = abs(f_star[j] - f_minus[j])
            if denom == 0: denom = 1
            dist = weights[j] * abs(f_star[j] - matrix[i, j]) / denom
            dists.append(dist)
        s_i[i] = sum(dists)
        r_i[i] = max(dists)
    s_star, s_minus_val = np.min(s_i), np.max(s_i)
    r_star, r_minus_val = np.min(r_i), np.max(r_i)
    denom_s = s_minus_val - s_star if s_minus_val != s_star else 1
    denom_r = r_minus_val - r_star if r_minus_val != r_star else 1
    q_i = v * (s_i - s_star) / denom_s + (1 - v) * (r_i - r_star) / denom_r
    return s_i, r_i, q_i

def calculate_ahp_vector(matrix, custom_ri_val):
    matrix = np.array(matrix, dtype=float)
    n = matrix.shape[0]
    col_sums = matrix.sum(axis=0)
    norm_matrix = np.divide(matrix, col_sums, where=col_sums!=0)
    weights = norm_matrix.mean(axis=1)
    lambda_max = np.dot(col_sums, weights)
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0
    cr = ci / custom_ri_val if custom_ri_val != 0 else 0
    return weights, cr, ci, lambda_max

# ==========================================
# INTERFACE - SIDEBAR (TOUS LES RÉGLAGES)
# ==========================================

st.sidebar.title("⚙️ Paramètres & Réglages")

# Choix du Mode Global
mode_principal = st.sidebar.radio(
    "1. Mode de Résolution",
    ["Matrice de Décision", "AHP (Comparaisons)"]
)

# ------------------------------------------
# MODE A : MATRICE DE DÉCISION
# ------------------------------------------
if mode_principal == "Matrice de Décision":
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Dimensions")
    num_alt = st.sidebar.number_input("Nombre d'alternatives", 2, 50, 4)
    num_crit = st.sidebar.number_input("Nombre de critères", 2, 50, 4)
    
    alt_names = [f"A{i+1}" for i in range(num_alt)]
    crit_names = [f"C{i+1}" for i in range(num_crit)]

    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Types de Critères")
    crit_types = []
    # Création d'un expander pour ne pas encombrer si n est grand
    with st.sidebar.expander("Définir Bénéfice/Coût", expanded=True):
        for i in range(num_crit):
            t = st.selectbox(
                f"{crit_names[i]}", 
                ["Bénéfice (+)", "Coût (-)"], 
                key=f"type_{i}"
            )
            crit_types.append(t)

    st.sidebar.markdown("---")
    st.sidebar.subheader("4. Pondération")
    methode_poids = st.sidebar.selectbox(
        "Méthode de calcul des poids",
        ["Saisie Manuelle", "Entropie (Calculé)", "CRITIC (Calculé)"]
    )
    
    weights_input = None
    if methode_poids == "Saisie Manuelle":
        with st.sidebar.expander("Saisir les poids", expanded=True):
            w_list = []
            for i in range(num_crit):
                val = st.number_input(f"Poids {crit_names[i]}", 0.0, 1.0, 1.0/num_crit, step=0.01, format="%.3f")
                w_list.append(val)
            # Normalisation automatique pour éviter les erreurs
            total = sum(w_list)
            if total > 0:
                weights_input = np.array(w_list) / total
            else:
                weights_input = np.ones(num_crit) / num_crit

    st.sidebar.markdown("---")
    st.sidebar.subheader("5. Méthode de Classement")
    method_rank = st.sidebar.selectbox(
        "Algorithme", 
        ["WASPAS", "TOPSIS", "VIKOR"]
    )
    
    # Paramètres Spécifiques (Lambda / v)
    lmbda, v_val = 0.5, 0.5
    
    if method_rank == "WASPAS":
        st.sidebar.markdown("#### Réglage WASPAS")
        lmbda = st.sidebar.slider("Paramètre Lambda (λ)", 0.0, 1.0, 0.5, 0.1)
        st.sidebar.latex(r"Q = \lambda Q^{(1)} + (1-\lambda)Q^{(2)}")
        
    elif method_rank == "VIKOR":
        st.sidebar.markdown("#### Réglage VIKOR")
        v_val = st.sidebar.slider("Stratégie v (Poids Utilité)", 0.0, 1.0, 0.5, 0.1)
        st.sidebar.caption("v > 0.5 : Vote majoritaire")


# ------------------------------------------
# MODE B : AHP
# ------------------------------------------
else:
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Dimensions AHP")
    num_alt_ahp = st.sidebar.number_input("Nb Alternatives", 2, 20, 3, key="ahp_na")
    num_crit_ahp = st.sidebar.number_input("Nb Critères", 2, 20, 3, key="ahp_nc")
    
    alts_ahp = [f"A{i+1}" for i in range(num_alt_ahp)]
    crits_ahp = [f"C{i+1}" for i in range(num_crit_ahp)]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Réglage RI (Index Aléatoire)")
    
    # Valeurs par défaut selon le tableau
    def_ri_c = RI_DICT_STD.get(num_crit_ahp, 1.56)
    def_ri_a = RI_DICT_STD.get(num_alt_ahp, 1.56)
    
    ri_crit_input = st.sidebar.number_input(
        f"RI Critères (n={num_crit_ahp})", 
        value=float(def_ri_c), format="%.2f"
    )
    ri_alt_input = st.sidebar.number_input(
        f"RI Alternatives (n={num_alt_ahp})", 
        value=float(def_ri_a), format="%.2f"
    )

# ==========================================
# ZONE PRINCIPALE - AFFICHAGE ET DONNÉES
# ==========================================

st.title("🤔 Application Aide à la Décision 🤔")

# >>>>>>>>>> LOGIQUE MATRICE DE DECISION <<<<<<<<<<
if mode_principal == "Matrice de Décision":
    
    # 1. Gestion de la Matrice (Session State)
    if 'dm_matrix' not in st.session_state:
        st.session_state.dm_matrix = pd.DataFrame(
            np.random.randint(1, 10, (num_alt, num_crit)), 
            index=alt_names, columns=crit_names
        )
    
    # Vérif redimensionnement
    if st.session_state.dm_matrix.shape != (num_alt, num_crit):
         st.session_state.dm_matrix = pd.DataFrame(
            np.zeros((num_alt, num_crit)), 
            index=alt_names, columns=crit_names
        )
         
    st.subheader("1. Saisie des Données (Matrice de Décision)")
    edited_df = st.data_editor(st.session_state.dm_matrix, key="editor_dm", use_container_width=True)

    # 2. Calcul des Poids
    weights = np.ones(num_crit) / num_crit
    error_msg = None
    
    if methode_poids == "Saisie Manuelle" and weights_input is not None:
        weights = weights_input
    elif methode_poids == "Entropie (Calculé)":
        try: weights = calculate_entropy_weights(edited_df)
        except Exception as e: error_msg = str(e)
    elif methode_poids == "CRITIC (Calculé)":
        try: weights = calculate_critic_weights(edited_df, crit_types)
        except Exception as e: error_msg = str(e)
        
    st.subheader("2. Poids Calculés / Saisis")
    if error_msg:
        st.error(f"Erreur de calcul des poids : {error_msg}")
    else:
        # Affichage Joli des Poids
        w_df = pd.DataFrame(weights.reshape(1, -1), columns=crit_names, index=["Poids"])
        st.dataframe(w_df.style.format("{:.4f}").background_gradient(axis=1, cmap="Blues"), use_container_width=True)

    # 3. Résultats Classement
    st.divider()
    st.subheader(f"3. Résultats : Méthode {method_rank}")
    
    if st.button("Lancer le Calcul", type="primary"):
        res_df = pd.DataFrame(index=alt_names)
        
        if method_rank == "WASPAS":
            wsm, wpm, qi = run_waspas(edited_df, weights, crit_types, lmbda)
            res_df["WSM (Somme)"] = wsm
            res_df["WPM (Produit)"] = wpm
            res_df["Score WASPAS"] = qi
            res_df["Rang"] = res_df["Score WASPAS"].rank(ascending=False).astype(int)
            st.dataframe(res_df.style.highlight_max(subset=["Score WASPAS"], color='#d1e7dd').format("{:.4f}"), use_container_width=True)
            
        elif method_rank == "TOPSIS":
            scores = run_topsis(edited_df, weights, crit_types)
            res_df["Score TOPSIS"] = scores
            res_df["Rang"] = res_df["Score TOPSIS"].fillna(-1).rank(ascending=False).astype(int)
            st.dataframe(res_df.style.highlight_max(subset=["Score TOPSIS"], color='#d1e7dd').format("{:.4f}"), use_container_width=True)
            
        elif method_rank == "VIKOR":
            s, r, q = run_vikor(edited_df, weights, crit_types, v_val)
            res_df["S (Utilité)"] = s
            res_df["R (Regret)"] = r
            res_df["Q (VIKOR)"] = q
            res_df["Rang"] = res_df["Q (VIKOR)"].rank(ascending=True).astype(int)
            st.dataframe(res_df.style.highlight_min(subset=["Q (VIKOR)"], color='#d1e7dd').format("{:.4f}"), use_container_width=True)

# >>>>>>>>>> LOGIQUE AHP <<<<<<<<<<
else:
    st.subheader("1. Matrices de Comparaison")
    
    # Matrice Critères
    st.markdown("**Comparaison des Critères**")
    if 'crit_ahp' not in st.session_state:
        st.session_state.crit_ahp = pd.DataFrame(np.eye(num_crit_ahp), index=crits_ahp, columns=crits_ahp)
    if st.session_state.crit_ahp.shape != (num_crit_ahp, num_crit_ahp):
        st.session_state.crit_ahp = pd.DataFrame(np.eye(num_crit_ahp), index=crits_ahp, columns=crits_ahp)
        
    crit_mat = st.data_editor(st.session_state.crit_ahp, key="ahp_crit_ed", use_container_width=True)
    
    # Calcul Poids Critères
    w_crit, cr_crit, ci_crit, _ = calculate_ahp_vector(crit_mat.values, ri_crit_input)
    
    # Affichage CR Critères
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1: st.info(f"Poids Critères calculés (Somme=1)")
    with c2: st.metric("CR Critères", f"{cr_crit:.4f}", delta_color="inverse" if cr_crit < 0.1 else "normal")
    if cr_crit > 0.1: st.error("Inconsistant (>0.1)")

    # Matrices Alternatives
    st.markdown("**Comparaison des Alternatives (par Critère)**")
    local_weights = {}
    tabs = st.tabs(crits_ahp)
    
    for i, crit in enumerate(crits_ahp):
        with tabs[i]:
            key = f"ahp_mat_{crit}"
            if key not in st.session_state:
                st.session_state[key] = pd.DataFrame(np.eye(num_alt_ahp), index=alts_ahp, columns=alts_ahp)
            if st.session_state[key].shape != (num_alt_ahp, num_alt_ahp):
                st.session_state[key] = pd.DataFrame(np.eye(num_alt_ahp), index=alts_ahp, columns=alts_ahp)
                
            mat = st.data_editor(st.session_state[key], key=f"ed_{key}", use_container_width=True)
            w_loc, cr_loc, _, _ = calculate_ahp_vector(mat.values, ri_alt_input)
            local_weights[crit] = w_loc
            
            if cr_loc > 0.1: st.warning(f"Attention : CR = {cr_loc:.3f} (> 0.1)")
            else: st.success(f"Consistance OK (CR = {cr_loc:.3f})")

    st.divider()
    if st.button("Calculer Résultat Final AHP", type="primary"):
        df_loc = pd.DataFrame(local_weights, index=alts_ahp)
        scores = df_loc.dot(w_crit)
        
        res_ahp = pd.DataFrame(scores, columns=["Score Global"])
        res_ahp["Rang"] = res_ahp["Score Global"].rank(ascending=False).astype(int)
        
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.write("Détail des Priorités Locales :")
            st.dataframe(df_loc.style.format("{:.4f}"), use_container_width=True)
        with col_res2:
            st.write("Classement Final :")
            st.dataframe(res_ahp.style.highlight_max(subset=["Score Global"], color='#d1e7dd').format("{:.4f}"), use_container_width=True)
            st.success(f"🏆 Vainqueur : **{res_ahp['Score Global'].idxmax()}**")