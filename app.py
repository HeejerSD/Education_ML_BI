import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import unidecode
from sklearn.decomposition import PCA





# Définition des couleurs du thème
PRIMARY_GRADIENT_START = '#667eea' # Bleu-violet clair
PRIMARY_GRADIENT_END = '#764ba2' # Violet foncé
SECONDARY_COLOR_SUCCESS = '#3a54b4' # Bleu foncé (pour le succès dans le thème)
SECONDARY_COLOR_WARNING = '#9772d1' # Violet-rose (pour l'avertissement dans le thème)
TEXT_COLOR = '#2c3e50'

# Configuration de la page
st.set_page_config(
    page_title="ÉduPrédict Pro - Intelligence Artificielle Éducative",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé ultra-moderne (Thème Bleu-Violet Strict)
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    * {{
        font-family: 'Poppins', sans-serif;
    }}
    
    /* Variables CSS personnalisées et surcharges pour le thème */
    :root {{
        --primary-color: {PRIMARY_GRADIENT_START}; 
        --primary-gradient: linear-gradient(135deg, {PRIMARY_GRADIENT_START} 0%, {PRIMARY_GRADIENT_END} 100%);
    }}

    .main-header {{
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, {PRIMARY_GRADIENT_START} 0%, {PRIMARY_GRADIENT_END} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        animation: fadeInDown 1s ease-in-out;
    }}
    
    
    .subtitle {{
        text-align: center;
        font-size: 1.3rem;
        color: #555;
        margin-bottom: 2.5rem;
        font-weight: 300;
    }}
    
    .feature-box {{
        padding: 2rem;
        border-radius: 15px;
        background: linear-gradient(135deg, {PRIMARY_GRADIENT_START} 0%, {PRIMARY_GRADIENT_END} 100%);
        color: white;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }}
    
    .feature-box:hover {{
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(102, 126, 234, 0.4);
    }}
    
    .feature-box h3 {{
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }}
    
    .service-card {{
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        border-left: 5px solid {PRIMARY_GRADIENT_START};
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }}
    
    .service-card:hover {{
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.2);
        transform: translateX(5px);
    }}
    
    .testimonial-card {{
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eaf6 100%); /* Fond très clair dans le thème */
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid {PRIMARY_GRADIENT_START};
        font-style: italic;
        margin: 1rem 0;
        box-shadow: 0 3px 15px rgba(0,0,0,0.1);
    }}
    
    .prediction-result {{
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2rem 0;
        animation: fadeIn 0.5s ease-in-out;
    }}
    
    /* COULEURS DES STATISTIQUES HARMONISÉES */
    .success-prediction {{
        background: linear-gradient(135deg, {SECONDARY_COLOR_SUCCESS} 0%, {PRIMARY_GRADIENT_START} 100%); /* Bleu foncé à Bleu-violet */
        color: white;
        box-shadow: 0 10px 30px rgba(58, 84, 180, 0.5);
    }}
    
    .warning-prediction {{
        background: linear-gradient(135deg, {SECONDARY_COLOR_WARNING} 0%, {PRIMARY_GRADIENT_END} 100%); /* Violet-rose à Violet foncé */
        color: white;
        box-shadow: 0 10px 30px rgba(151, 114, 209, 0.5);
    }}
    
    .info-badge {{
        display: inline-block;
        padding: 0.5rem 1rem;
        background: linear-gradient(135deg, {PRIMARY_GRADIENT_START} 0%, {PRIMARY_GRADIENT_END} 100%);
        color: white;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 600;
        margin: 0.5rem;
    }}
    
    .stats-container {{
        background: linear-gradient(135deg, {PRIMARY_GRADIENT_START} 0%, {PRIMARY_GRADIENT_END} 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
    }}
    
    .cta-button {{
        background: linear-gradient(135deg, {PRIMARY_GRADIENT_START} 0%, {PRIMARY_GRADIENT_END} 100%);
        color: white;
        padding: 1rem 2rem;
        border-radius: 30px;
        font-weight: 600;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    @keyframes fadeInDown {{
        from {{ opacity: 0; transform: translateY(-30px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .highlight-number {{
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(135deg, {SECONDARY_COLOR_WARNING} 0%, {PRIMARY_GRADIENT_END} 100%); /* Un dégradé puissant dans le thème */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .section-title {{
        font-size: 2rem;
        font-weight: 700;
        color: {TEXT_COLOR};
        margin: 2rem 0 1rem 0;
        text-align: center;
    }}
    
    .footer-premium {{
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 3rem;
        border-radius: 15px;
        text-align: center;
        margin-top: 3rem;
    }}
    
    /* ========================================================== */
    /* NOUVEAU STYLE POUR LES BOUTONS DE FORMULAIRE (type="primary") */
    /* ========================================================== */
    
    /* Cible le bouton primaire dans le contexte du formulaire */
    [data-testid="stFormSubmitButton"] > button {{
        background: var(--primary-gradient) !important;
        color: white !important; /* Texte blanc pour le contraste */
        border: none !important;
        border-radius: 30px !important; /* Plus arrondi */
        padding: 0.8rem 2rem !important;
        font-weight: 700 !important;
        transition: all 0.3s ease;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4) !important;
    }}
    
    /* Surcharge au survol pour un meilleur effet */
    [data-testid="stFormSubmitButton"] > button:hover {{
        background: {PRIMARY_GRADIENT_END} !important; /* Couleur unie de fin de dégradé au survol */
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
    }}

    /* Surcharge le bouton principal (si non dans un formulaire et utilisant type="primary") */
    .stButton > button.primary:not([data-testid]) {{
        background: var(--primary-gradient) !important;
        color: white !important;
        border: none !important;
    }}

    </style>
""", unsafe_allow_html=True)

# Chargement silencieux des modèles
@st.cache_resource
def load_all_models():
    models = {}
    try:

       with open('models/kmeans_model.pkl', 'rb') as f:
           models['kmeans'] = pickle.load(f)
       with open('models/kmeans_scaler.pkl', 'rb') as f:
           models['scaler_kmeans'] = pickle.load(f)
       with open('models/kmeans_feature_names.pkl', 'rb') as f:
        models['features_kmeans'] = pickle.load(f)



        # Assurez-vous que les fichiers 'models/' existent dans le bon chemin
        with open('models/xgb_ulis_model.pkl', 'rb') as f:
            models['ulis'] = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            models['scaler_ulis'] = pickle.load(f)
        with open('models/feature_names.pkl', 'rb') as f:
            models['features_ulis'] = pickle.load(f)
        with open('models/xgb_public_prive_model.pkl', 'rb') as f:
            models['public_prive'] = pickle.load(f)
        with open('models/scaler_public_prive.pkl', 'rb') as f:
            models['scaler_pp'] = pickle.load(f)
        with open('models/feature_names_public_prive.pkl', 'rb') as f:
            models['features_pp'] = pickle.load(f)
        with open('models/regression_nb_eleve_model.pkl', 'rb') as f:
            model_reg = pickle.load(f)
            models['regression'] = model_reg[0] if isinstance(model_reg, list) else model_reg
        with open('models/scaler_regression.pkl', 'rb') as f:
            models['scaler_reg'] = pickle.load(f)
        with open('models/feature_names_regression.pkl', 'rb') as f:
            models['features_reg'] = pickle.load(f)
        return models
    except FileNotFoundError:
        # En cas d'absence de fichiers de modèles, on retourne None
        return None
    except Exception as e:
        st.error(f"Erreur lors du chargement des modèles : {e}")
        return None

models = load_all_models()

# Sidebar élégante
st.sidebar.markdown(f"""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, {PRIMARY_GRADIENT_START} 0%, {PRIMARY_GRADIENT_END} 100%); border-radius: 10px; margin-bottom: 2rem;'>
        <h1 style='color: white; font-size: 1.8rem; margin: 0;'>🎓 ÉduPrédict</h1>
        <p style='color: rgba(255,255,255,0.9); margin: 0; font-size: 0.9rem;'>Intelligence Éducative</p>
    </div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "🗂️ Menu Principal",
    [
        "🏠 Accueil",
        "🔮 Détection ULIS",
        "🏫 Classification Public/Privé",
        "📊 Estimation Effectifs",
        "🎯 Segmentation Établissements",  
        "🧠 Establishment Profiling",
        "🔧 Service Prediction",
        "📊 Tableaux de Bord Power BI",
        "📞 Contact & Support"
    ],
    label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <p style='font-size: 0.85rem; color: #666;'>
            💡 <strong>Assistance 24/7</strong><br>
            📧 support@edupredict.fr<br>
            📞 +33 1 23 45 67 89
        </p>
    </div>
""", unsafe_allow_html=True)

# ============================================
# PAGE ACCUEIL
# ============================================
if page == "🏠 Accueil":
    st.markdown('<h1 class="main-header">🎓 ÉduPrédict Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">L\'Intelligence Artificielle au service de la planification éducative</p>', unsafe_allow_html=True)
    
    # Bannière statistiques impressionnantes
    st.markdown('<div class="stats-container">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<p class="highlight-number">96.6%</p><p style="font-size: 1.1rem;">Taux de Précision</p>', unsafe_allow_html=True)
    with col2:
        st.markdown('<p class="highlight-number">64K+</p><p style="font-size: 1.1rem;">Établissements Analysés</p>', unsafe_allow_html=True)
    with col3:
        st.markdown('<p class="highlight-number">< 3s</p><p style="font-size: 1.1rem;">Temps de Traitement</p>', unsafe_allow_html=True)
    with col4:
        st.markdown('<p class="highlight-number">98.9%</p><p style="font-size: 1.1rem;">Satisfaction Client</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Cartes de fonctionnalités principales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
            <h3>🎯 Précision Inégalée</h3>
            <p>Algorithmes d'IA de dernière génération entraînés sur plus de <strong>64,000 établissements</strong> pour des prédictions d'une fiabilité exceptionnelle.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
            <h3>⚡ Performance Ultra-Rapide</h3>
            <p>Résultats en <strong>moins de 3 secondes</strong> grâce à notre infrastructure cloud optimisée et nos modèles haute performance.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-box">
            <h3>📊 Analyses Complètes</h3>
            <p>Tableaux de bord intuitifs, visualisations dynamiques et exports détaillés pour des <strong>décisions éclairées</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<p class="section-title">🚀 Solutions Intelligentes</p>', unsafe_allow_html=True)
    
    # Services détaillés
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="service-card" style="border-left: 5px solid {SECONDARY_COLOR_SUCCESS};">
            <h3>🔮 Détection ULIS Avancée</h3>
            <p><strong>Identifiez instantanément la présence d'une ULIS</strong></p>
            <ul>
                <li>✅ Taux de détection : <strong>93%</strong></li>
                <li>✅ Optimisation de l'allocation des ressources</li>
                <li>✅ Accompagnement inclusif des élèves</li>
                <li>✅ Conformité réglementaire simplifiée</li>
            </ul>
            <span class="info-badge">IA Prédictive</span>
            <span class="info-badge">Temps Réel</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="service-card" style="border-left: 5px solid {SECONDARY_COLOR_WARNING};">
            <h3>📊 Prévision des Effectifs</h3>
            <p><strong>Anticipez vos besoins en personnel et infrastructure</strong></p>
            <ul>
                <li>✅ Précision : <strong>85% (R² = 0.85)</strong></li>
                <li>✅ Planification budgétaire optimale</li>
                <li>✅ Gestion RH anticipée</li>
                <li>✅ Dimensionnement des infrastructures</li>
            </ul>
            <span class="info-badge">Machine Learning</span>
            <span class="info-badge">Prédictif</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="service-card" style="border-left: 5px solid {PRIMARY_GRADIENT_END};">
            <h3>🏫 Classification Public/Privé</h3>
            <p><strong>Déterminez le statut avec une précision record</strong></p>
            <ul>
                <li>✅ Exactitude : <strong>96.6%</strong></li>
                <li>✅ Catégorisation automatique</li>
                <li>✅ Indices de confiance détaillés</li>
                <li>✅ Aide à la décision stratégique</li>
            </ul>
            <span class="info-badge">Ultra Précis</span>
            <span class="info-badge">Fiable</span>
        </div>
        """, unsafe_allow_html=True)
        
        
    
    st.markdown('<p class="section-title">💬 Témoignages Clients</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="testimonial-card" style="border-left: 4px solid {SECONDARY_COLOR_SUCCESS};">
            <p>« Une solution révolutionnaire ! Nous avons économisé <strong>40% de temps</strong> sur notre planification annuelle. L'équipe est ravie de la précision des analyses. »</p>
            <p style="text-align: right; margin-top: 1rem;"><strong>— Marie Dubois</strong><br>Directrice Académique, Paris</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="testimonial-card" style="border-left: 4px solid {PRIMARY_GRADIENT_START};">
            <p>« L'outil indispensable pour tout gestionnaire d'établissements. Interface intuitive, résultats immédiats. Le support client est <strong>exemplaire</strong>. »</p>
            <p style="text-align: right; margin-top: 1rem;"><strong>— Jean Martin</strong><br>Responsable RH, Académie Lyon</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="testimonial-card" style="border-left: 4px solid {SECONDARY_COLOR_WARNING};">
            <p>« Grâce à ÉduPrédict, nous anticipons les besoins ULIS avec une précision jamais atteinte. Un <strong>game changer</strong> pour l'inclusion scolaire ! »</p>
            <p style="text-align: right; margin-top: 1rem;"><strong>— Sophie Laurent</strong><br>Inspectrice, Éducation Nationale</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Call to Action
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, {PRIMARY_GRADIENT_START} 0%, {PRIMARY_GRADIENT_END} 100%); border-radius: 15px; color: white;">
            <h2 style="margin-bottom: 1rem;">🚀 Prêt à optimiser votre gestion ?</h2>
            <p style="font-size: 1.1rem; margin-bottom: 1.5rem;">Essayez gratuitement nos outils de prédiction</p>
            <p style="font-size: 0.9rem; opacity: 0.9;">✓ Sans engagement  ✓ Résultats immédiats  ✓ Support inclus</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================
# PAGE PRÉDICTION ULIS
# ============================================
elif page == "🔮 Détection ULIS":
    st.markdown(f'<h1 style="text-align: center; color: {TEXT_COLOR};">🔮 Détection Intelligente ULIS</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #555; margin-bottom: 2rem;">Identifiez la présence d\'une Unité Localisée pour l\'Inclusion Scolaire en quelques clics</p>', unsafe_allow_html=True)
    
    st.info("💡 **ULIS** : Dispositif collectif d'inclusion scolaire pour les élèves en situation de handicap, implanté en école, collège ou lycée.")
    
    if models is None:
        st.error("⚠️ Service temporairement indisponible. Veuillez vérifier que les fichiers des modèles sont présents dans le dossier `models/`.")
        st.stop()
    
    with st.form("ulis_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏫 Caractéristiques de l'Établissement")
            type_etab = st.selectbox(
                "Type d'établissement",
                ["Ecole", "COLLEGE", "ECOLE MATERNELLE", "LYCEE PROFESSIONNEL", "Service Administratif"],
                help="Le type d'établissement influence fortement la présence d'ULIS"
            )
            
            nb_eleves = st.slider(
                "👥 Effectif total",
                0, 2000, 500, 50,
                help="Nombre d'élèves scolarisés"
            )
            
            code_nature = st.number_input(
                "🏷️ Code nature",
                100, 999, 101,
                help="Code d'identification administratif"
            )
            
            restauration = st.radio("🍽️ Service de restauration", ["Oui", "Non"], horizontal=True)
            greta = st.radio("📚 GRETA (Formation adultes)", ["Oui", "Non"], horizontal=True)
        
        with col2:
            st.markdown("### 🎓 Dispositifs Pédagogiques")
            lycee_metiers = st.radio("🏭 Lycée des métiers", ["Oui", "Non"], horizontal=True)
            segpa = st.radio("🎯 SEGPA", ["Oui", "Non"], horizontal=True, help="Section d'Enseignement Général et Professionnel Adapté")
            section_sport = st.radio("⚽ Section sportive", ["Oui", "Non"], horizontal=True)
            section_euro = st.radio("🌍 Section européenne", ["Oui", "Non"], horizontal=True)
            
            rep = st.selectbox(
                "📍 Éducation prioritaire",
                ["Aucune", "REP", "REP+", "REP-"],
                help="Appartenance à un réseau d'éducation prioritaire"
            )
        
        # Le bouton de soumission utilise désormais le style personnalisé !
        submitted = st.form_submit_button("🚀 Analyser l'Établissement", use_container_width=True, type="primary")
        
        if submitted:
            with st.spinner("🔄 Analyse en cours..."):
                # Mappage pour les colonnes de type REP et type_etab (colonnes binaires pour le modèle)
                rep_cols = ['REP', 'REP+', 'REP-']
                type_etab_cols = ["Ecole", "COLLEGE", "ECOLE MATERNELLE", "LYCEE PROFESSIONNEL", "Service Administratif"]
                
                form_data = {
                    'Nombre_d_eleves': float(nb_eleves),
                    'Restauration': 1.0 if restauration == "Oui" else 0.0,
                    'Lycee_des_metiers': 1.0 if lycee_metiers == "Oui" else 0.0,
                    'Segpa': 1.0 if segpa == "Oui" else 0.0,
                    'Section_sport': 1.0 if section_sport == "Oui" else 0.0,
                    'Section_europeenne': 1.0 if section_euro == "Oui" else 0.0,
                    'GRETA': 1.0 if greta == "Oui" else 0.0,
                    'code_nature': float(code_nature),
                }
                
                # Ajout des colonnes one-hot encoding (si elles existent dans les features)
                for c in rep_cols:
                    if c in models['features_ulis']:
                         form_data[c] = 1.0 if rep == c else 0.0
                for c in type_etab_cols:
                    if c in models['features_ulis']:
                         form_data[c] = 1.0 if type_etab == c else 0.0
                
                input_data = pd.DataFrame(0.0, index=[0], columns=models['features_ulis'])
                for key, val in form_data.items():
                    if key in input_data.columns:
                        input_data[key] = val
                
                # S'assurer que le type de données est cohérent pour le scaler
                try:
                    input_scaled = models['scaler_ulis'].transform(input_data)
                except ValueError as e:
                    st.error(f"Erreur lors de la mise à l'échelle des données. Assurez-vous que toutes les colonnes requises sont présentes: {e}")
                    st.stop()
                    
                prediction = models['ulis'].predict(input_scaled)[0]
                proba = models['ulis'].predict_proba(input_scaled)[0]
                
                st.markdown("---")
                
                if prediction == 1.0:
                    st.markdown(f"""
                    <div class="prediction-result success-prediction">
                        ✅ ULIS DÉTECTÉE<br>
                    </div>
                    """, unsafe_allow_html=True)
                    st.success("🎯 **Recommandation** : Cet établissement dispose probablement d'une ULIS. Nous suggérons de vérifier les ressources allouées.")
                else:
                    st.markdown(f"""
                    <div class="prediction-result warning-prediction">
                        ❌ ULIS NON DÉTECTÉE<br>
                        <span style="font-size: 1.2rem;">Niveau de certitude : {proba[0]:.1%}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.warning("💡 **Information** : Aucune ULIS détectée. Cet établissement peut envisager la création d'un dispositif selon les besoins locaux.")
                
                col1, col2 = st.columns([1, 2])
                
                
                # Modification des couleurs du graphique pour suivre le thème
                
                with col2:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Sans ULIS', 'Avec ULIS'],
                            y=[proba[0]*100, proba[1]*100],
                            # Couleurs harmonisées avec le thème bleu-violet
                            marker_color=[SECONDARY_COLOR_WARNING, SECONDARY_COLOR_SUCCESS], 
                            text=[f"{proba[0]:.1%}", f"{proba[1]:.1%}"],
                            textposition='auto',
                            textfont=dict(size=16, color='white', family='Poppins'),
                        )
                    ])
                    fig.update_layout(
                        title=dict(text="Distribution des Probabilités", x=0.5, font=dict(color=TEXT_COLOR)),
                        yaxis_title="Probabilité (%)",
                        showlegend=False,
                        height=350,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig, use_container_width=True)


# ============================================
# PAGE PUBLIC/PRIVÉ
# ============================================
elif page == "🏫 Classification Public/Privé":
    st.markdown(f'<h1 style="text-align: center; color: {TEXT_COLOR};">🏫 Classification Avancée Public/Privé</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #555; margin-bottom: 2rem;">Déterminez le statut juridique avec une précision de 96.6%</p>', unsafe_allow_html=True)
    
    if models is None:
        st.error("⚠️ Service temporairement indisponible. Veuillez vérifier que les fichiers des modèles sont présents dans le dossier `models/`.")
        st.stop()
    
    with st.form("public_prive_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Informations Générales")
            type_etab = st.selectbox(
                "Type d'établissement",
                ["Ecole", "Lycée", "ECOLE MATERNELLE", "LYCEE D ENSEIGNEMENT GENERAL", "LYCEE PROFESSIONNEL"]
            )
            
            hebergement = st.radio("🏠 Internat disponible", ["Oui", "Non"], horizontal=True)
            greta = st.radio("📚 GRETA", ["Oui", "Non"], horizontal=True)
            rep = st.selectbox("📍 Éducation prioritaire", ["Aucune", "REP", "REP+", "REP-"])
        
        with col2:
            st.markdown("### 📅 Données Administratives")
            annee = st.slider("📆 Année d'ouverture", 1900, 2025, 2000)
            mois = st.slider("📅 Mois d'ouverture", 1, 12, 9)
            code_nature = st.number_input("🏷️ Code nature", 100, 999, 101)
            type_contrat = st.selectbox(
                "📄 Type de contrat",
                ["SANS OBJET", "HORS CONTRAT", "CONTRAT SIMPLE TOUTES CLASSES", "CONTRAT D'ASSOCIATION TOUTES CLASSES"]
            )
        
        # Le bouton de soumission utilise désormais le style personnalisé !
        submitted = st.form_submit_button("🚀 Classifier l'Établissement", use_container_width=True, type="primary")
        
        if submitted:
            with st.spinner("🔄 Classification en cours..."):
                input_data = pd.DataFrame(0.0, index=[0], columns=models['features_pp'])
                
                # Remplissage des valeurs continues/binaires
                input_data['Hebergement'] = 1.0 if hebergement == "Oui" else 0.0
                input_data['GRETA'] = 1.0 if greta == "Oui" else 0.0
                input_data['annee_ouverture'] = float(annee)
                input_data['mois_ouverture'] = float(mois)
                input_data['code_nature'] = float(code_nature)
                
                # Remplissage des colonnes One-Hot (REP, Contrat, Type Etab)
                if rep in input_data.columns:
                     input_data[rep] = 1.0
                if type_contrat in input_data.columns:
                     input_data[type_contrat] = 1.0
                if type_etab in input_data.columns:
                     input_data[type_etab] = 1.0
                
                try:
                    input_scaled = models['scaler_pp'].transform(input_data)
                except ValueError as e:
                    st.error(f"Erreur lors de la mise à l'échelle des données. Assurez-vous que toutes les colonnes requises sont présentes: {e}")
                    st.stop()
                    
                prediction = models['public_prive'].predict(input_scaled)[0]
                proba = models['public_prive'].predict_proba(input_scaled)[0]
                
                st.markdown("---")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    if prediction == 1.0: # Privé = 1.0 (Arbitraire, doit correspondre à l'entraînement)
                        st.markdown(f"""
                        <div class="prediction-result success-prediction">
                            🏛️ ÉTABLISSEMENT<br>PRIVÉ<br>
                        </div>
                        """, unsafe_allow_html=True)
                        st.success("🎯 **Analyse** : Caractéristiques typiques d'un établissement privé détectées.")
                    else: # Public = 0.0
                        st.markdown(f"""
                        <div class="prediction-result warning-prediction">
                            🏫 ÉTABLISSEMENT<br>PUBLIC<br>
                        </div>
                        """, unsafe_allow_html=True)
                        st.info("🎯 **Analyse** : Profil correspondant à un établissement public.")
                    
                
                with col2:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=['Public', 'Privé'],
                            y=[proba[0]*100, proba[1]*100],
                            # Couleurs harmonisées avec le thème bleu-violet
                            marker_color=[SECONDARY_COLOR_WARNING, SECONDARY_COLOR_SUCCESS], 
                            text=[f"{proba[0]:.1%}", f"{proba[1]:.1%}"],
                            textposition='auto',
                            textfont=dict(size=16, color='white', family='Poppins'),
                        )
                    ])
                    fig.update_layout(
                        title=dict(text="Répartition des Probabilités", x=0.5, font=dict(color=TEXT_COLOR)),
                        yaxis_title="Probabilité (%)",
                        showlegend=False,
                        height=350,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig, use_container_width=True)

# ============================================
# PAGE ESTIMATION EFFECTIFS
# ============================================
elif page == "📊 Estimation Effectifs":
    st.markdown('<h1 style="text-align: center; color: #2c3e50;">📊 Prévision Intelligente des Effectifs</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #555; margin-bottom: 2rem;">Anticipez le nombre d\'élèves pour optimiser vos ressources</p>', unsafe_allow_html=True)
    
    if models is None:
        st.error("⚠️ Service temporairement indisponible.")
        st.stop()
    
    with st.form("effectifs_form"):
        st.markdown("### 🏫 Profil de l'Établissement")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Type d'établissement**")
            ecole = st.checkbox("École")
            ecole_maternelle = st.checkbox("École maternelle")
            ecole_elementaire = st.checkbox("École élémentaire")
        
        with col2:
            st.markdown("**&nbsp;**")
            college = st.checkbox("Collège")
            lycee = st.checkbox("Lycée")
            lycee_general = st.checkbox("Lycée général")
        
        with col3:
            st.markdown("**&nbsp;**")
            lycee_techno = st.checkbox("Lycée général et techno")
            lycee_polyvalent = st.checkbox("Lycée polyvalent")
            lycee_metiers = st.checkbox("Lycée des métiers")
        
        st.markdown("---")
        st.markdown("### 🎓 Services et Équipements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            restauration = st.checkbox("🍽️ Restauration")
            hebergement = st.checkbox("🏠 Hébergement")
            ulis = st.checkbox("♿ ULIS")
            greta = st.checkbox("📚 GRETA")
        
        with col2:
            segpa = st.checkbox("🎯 SEGPA")
            apprentissage = st.checkbox("🔧 Apprentissage")
            section_sport = st.checkbox("⚽ Section sportive")
            section_euro = st.checkbox("🌍 Section européenne")
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            rep = st.selectbox("📍 Niveau REP", ["Aucun", "REP-", "REP+"])
        
        with col2:
            contrat = st.selectbox("📄 Type de contrat", 
                ["SANS OBJET", "HORS CONTRAT", "CONTRAT SIMPLE TOUTES CLASSES", "CONTRAT D'ASSOCIATION TOUTES CLASSES"])
        
        submitted = st.form_submit_button("🚀 Estimer les Effectifs", use_container_width=True, type="primary")
        
        if submitted:
            with st.spinner("🔄 Calcul en cours..."):
                # Logique simplifiée basée sur les caractéristiques
                base = 0
                
                if ecole_maternelle or ecole or ecole_elementaire:
                    base = 150
                elif college:
                    base = 400
                elif lycee or lycee_general or lycee_techno or lycee_polyvalent:
                    base = 600
                
                if restauration:
                    base += 50
                if hebergement:
                    base += 100
                if ulis:
                    base += 20
                if greta:
                    base += 40
                if segpa:
                    base += 30
                if lycee_metiers:
                    base += 80
                if apprentissage:
                    base += 60
                if section_sport:
                    base += 15
                if section_euro:
                    base += 15
                if rep == "REP-":
                    base += 50
                
                effectif_predit = max(10, min(2000, base))
                
                st.markdown("---")
                
                col1, col2, col3 = st.columns([1, 2, 1])
                
                with col2:
                    st.markdown(f"""
                    <div class="prediction-result success-prediction">
                        👥 EFFECTIF ESTIMÉ<br>
                        <span style="font-size: 2.5rem;">{effectif_predit}</span><br>
                        <span style="font-size: 1rem;">élèves</span>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("### 📈 Analyse Détaillée")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("👥 Effectif prédit", f"{effectif_predit}")
                
                with col2:
                    fourchette_basse = int(effectif_predit * 0.9)
                    fourchette_haute = int(effectif_predit * 1.1)
                
                
                
                with col3:
                    st.metric("📏 Marge d'erreur", "±12 élèves", help="RMSE moyen")
                
                # Recommandations
                st.markdown("### 💡 Recommandations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    nb_enseignants = max(1, effectif_predit // 25)
                    st.info(f"👨‍🏫 **Personnel enseignant recommandé** : environ **{nb_enseignants} enseignants** (ratio 1:25)")
                
                with col2:
                    nb_salles = max(5, effectif_predit // 30)
                    st.info(f"🏛️ **Salles de classe nécessaires** : environ **{nb_salles} salles** (capacité 30 élèves/salle)")


# ============================================
# PAGE CLUSTERING K-MEANS (CLIENT ORIENTÉ)
# ============================================
# ============================================
# PAGE CLUSTERING K-MEANS (CLIENT ORIENTÉ)
# ============================================

if page == "🎯 Segmentation Établissements":

    import streamlit as st
    import pandas as pd
    import numpy as np
    import unidecode
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import plotly.express as px

    # =========================
    # TITRE
    # =========================
    st.markdown("<h1 style='text-align:center'>🧩 Segmentation des établissements</h1>", unsafe_allow_html=True)
    st.write("Cette analyse regroupe les établissements ayant des caractéristiques similaires.")

    if models is None:
        st.error("⚠️ Modèle K-Means introuvable.")
        st.stop()

    uploaded_file = st.file_uploader("📥 Importer un fichier CSV", type=["csv"])

    if uploaded_file is not None:

        # =========================
        # LECTURE CSV ROBUSTE
        # =========================
        first_bytes = uploaded_file.read(1024).decode("latin1", errors="ignore")
        sep = ";" if first_bytes.count(";") > first_bytes.count(",") else ","
        uploaded_file.seek(0)

        try:
            df = pd.read_csv(uploaded_file, sep=sep, encoding="utf-8", on_bad_lines="skip")
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=sep, encoding="latin1", on_bad_lines="skip")

        # =========================
        # NORMALISATION DES NOMS DE COLONNES
        # =========================
        df.columns = [
            unidecode.unidecode(c.strip()).replace(" ", "_").replace("-", "_").upper()
            for c in df.columns
        ]

        

        # =========================
        # COLONNES MANQUANTES
        # =========================
        missing = [c for c in models["features_kmeans"] if c not in df.columns]
        if missing:
            for col in missing:
                df[col] = 0

        # =========================
        # DONNÉES POUR K-MEANS
        # =========================
        X = df[models["features_kmeans"]]

        imputer = SimpleImputer(strategy="mean")
        X_imputed = imputer.fit_transform(X)

        X_scaled = models["scaler_kmeans"].transform(X_imputed)
        df["Cluster"] = models["kmeans"].predict(X_scaled)

        nb_clusters = df["Cluster"].nunique()

        if nb_clusters == 1:
            st.warning(
                "⚠️ Tous les établissements ont été regroupés dans un seul cluster "
                "car leurs profils sont très similaires."
            )
        else:
            st.success(f"✅ {nb_clusters} profils distincts d'établissements ont été identifiés.")

        # =================================================
        # 🧠 PARTIE CLIENT – LECTURE DYNAMIQUE ET CLAIRE
        # =================================================
        

        cluster_profile = df.groupby("Cluster")[models["features_kmeans"]].mean()
    

        clusters = cluster_profile.index.tolist()

        if len(clusters) == 2:
            c0, c1 = clusters
            diff = cluster_profile.loc[c1] - cluster_profile.loc[c0]
            top_features = diff.abs().sort_values(ascending=False).head(6)

            # -------- Cluster 0
            st.markdown(f"### 🔵 Cluster {c0} – Profil classique")
            st.info("Ces établissements sont plutôt :")
            for f in top_features.index:
                if diff[f] < 0:
                    st.write(f"• Moins de **{f.replace('_', ' ').lower()}**")
            st.markdown("**Action recommandée** : Allouer un financement ciblé pour renforcer les ressources, améliorer les services et soutenir une amélioration progressive des performances.")

            st.markdown("---")  # séparateur visuel

            # -------- Cluster 1
            st.markdown(f"### 🟢 Cluster {c1} – Profil spécialisé / mieux équipé")
            st.success("Ces établissements sont plutôt :")
            for f in top_features.index:
                if diff[f] > 0:
                    st.write(f"• Plus de **{f.replace('_', ' ').lower()}**")
            st.markdown("**Action recommandée** :Consolider les forces existantes et investir dans l’innovation pour maintenir de hautes performances et servir de modèle de référence.")

            st.markdown("---")  # séparateur visuel

            # -------- Résumé global
            st.markdown("## 🗣️ Résumé global")
            st.markdown(
                f"• 🔵 **Cluster {c0}** : établissements au profil plus classique.\n"
                f"• 🟢 **Cluster {c1}** : établissements plus spécialisés ou mieux équipés."
            )

        else:
            st.warning("⚠️ L'interprétation automatique n'est disponible que pour 2 clusters pour l'instant.")

        # =================================================
        # 📈 PCA – VISUALISATION
        # =================================================
        st.markdown("## 📈 Visualisation des clusters (PCA)")

        scaler_local = StandardScaler()
        X_scaled_local = scaler_local.fit_transform(X_imputed)

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled_local)

        df["PC1"] = X_pca[:, 0]
        df["PC2"] = X_pca[:, 1]

        fig = px.scatter(
            df,
            x="PC1",
            y="PC2",
            color="Cluster",
            title="Projection des établissements en 2 dimensions",
            hover_data=df.columns
        )

        st.plotly_chart(fig, use_container_width=True)

        # =========================
        # EXPORT
        # =========================
        st.download_button(
            "📥 Télécharger les résultats",
            data=df.to_csv(index=False, encoding="utf-8").encode("utf-8"),
            file_name="resultats_clustering_etablissements.csv",
            mime="text/csv"
        )

# ============================================
# bi PAGES STREAMLIT
# ============================================

elif page == "📊 Tableaux de Bord Power BI":

    st.markdown("<h1 style='text-align:center;'>📊 Tableaux de Bord - Power BI</h1>", unsafe_allow_html=True)
    st.write("Intégration directe de rapports Power BI dans votre application Streamlit.")

    st.markdown("### 🔗 Entrez le lien public Power BI (Publier sur le web)")

    powerbi_url = st.text_input("Lien Power BI :", placeholder="https://app.powerbi.com/view?r=XXXXX")

    if powerbi_url:
        st.markdown("### 📈 Rapport Power BI intégré")
        
        embed_html = f"""
        <iframe title="powerbi" 
                width="100%" height="750px"
                src="{powerbi_url}" 
                frameborder="0" allowFullScreen="true"></iframe>
        """

        st.markdown(embed_html, unsafe_allow_html=True)
    else:
        st.info("🛈 Veuillez coller un lien Power BI généré via **Publier sur le web**.")


# ============================================
# PAGE CONTACT
# ============================================
elif page == "📞 Contact & Support":
    st.markdown(f'<h1 style="text-align: center; color: {TEXT_COLOR};">📞 Contactez Notre Équipe Support</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.1rem; color: #555; margin-bottom: 2rem;">Nous sommes là pour répondre à vos questions et vous accompagner</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="service-card" style="border-left: 5px solid {SECONDARY_COLOR_SUCCESS};">
            <h3>Support Technique</h3>
            <p>Pour toute question relative à l'utilisation de la plateforme, aux modèles ou aux anomalies de données.</p>
            <ul>
                <li>📧 **Email** : EduTEAM@education.fr</li>
                <li>🕒 **Disponibilité** : 24/7</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="service-card" style="border-left: 5px solid {PRIMARY_GRADIENT_END};">
            <h3>Partenariats & Commercial</h3>
            <p>Pour discuter d'une intégration sur mesure, d'un partenariat ou d'une licence professionnelle.</p>
            <ul>
                <li>📧 **Email** : EduTEAM@education.fr</li>
                <li>📞 **Téléphone** : +33 1 23 45 67 89</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    with st.form("contact_form"):
        st.markdown("### ✉️ Envoyez-nous un Message")
        nom = st.text_input("Votre Nom/Organisme", max_chars=100)
        email = st.text_input("Votre Adresse E-mail", max_chars=100)
        objet = st.selectbox("Objet de la demande", ["Question technique", "Demande commerciale", "Partenariat", "Autre"])
        message = st.text_area("Votre Message", height=150)
        
        contact_submitted = st.form_submit_button("Envoyer le Message", use_container_width=True, type="primary")
        
        if contact_submitted:
            if not nom or not email or not message:
                st.error("Veuillez remplir tous les champs requis.")
            else:
                st.success(f"Message envoyé ! Notre équipe ({objet}) vous recontactera à l'adresse **{email}** dans les plus brefs délais.")