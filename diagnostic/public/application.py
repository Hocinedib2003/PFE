import os
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots

os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'
os.environ['TORCH_DYNAMO_DISABLE'] = '1'
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import joblib
import torch
from torchvision import transforms
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="LMA Diagnostic Assistant",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

password = st.text_input("Entrez le mot de passe", type="password")

if password:
    if password == st.secrets.get("PASSWORD"):
        st.success("✅ Accès autorisé")
    else:
        st.error("❌ Accès refusé. Contactez l'administrateur.")
        st.stop()
else:
    st.warning("⛔ Veuillez entrer le mot de passe pour accéder à l'application.")
    st.stop()

st.markdown("""
<style>
    /* Style du titre principal */
    .header {
        font-size: 2.5rem !important;
        color: #2b5876 !important; /* Couleur bleu foncé */
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Style des boîtes de résultats */
    .result-box {
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); /* Ombre légère */
    }

    /* Style pour résultats "sains" */
    .healthy {
        background-color: #e6f7e6; /* Vert très clair */
        border-left: 5px solid #4CAF50; /* Bordure verte */
    }

    /* Style pour résultats "malades" */
    .sick {
        background-color: #ffebee; /* Rouge très clair */
        border-left: 5px solid #F44336; /* Bordure rouge */
    }

    /* Conteneur pour l'image segmentée */
    .segmentation-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
    }

    /* Style des boîtes de segmentation */
    .segmentation-box {
        width: 48%;
        border: 1px solid #ddd; /* Bordure grise */
        border-radius: 8px;
        padding: 1rem;
    }

    /* Style de la barre de progression */
    .stProgress > div > div > div > div {
        background-color: #2b5876; /* Bleu foncé */
    }

    /* Style du spinner */
    .stSpinner > div > div {
        border-top-color: #2b5876 !important; /* Bleu foncé */
    }

    /* Boîte de diagnostic */
    .diagnostic-box {
        transition: all 0.3s ease; /* Animation fluide */
        margin-bottom: 2rem;
    }

    /* Effet hover sur la boîte de diagnostic */
    .diagnostic-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
    }

    /* Style des onglets */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px 8px 0 0;
    }

    /* Style des tableaux de paramètres */
    .param-table {
        font-family: Arial, sans-serif;
        border-collapse: collapse;
        width: 100%;
    }

    .param-table td, .param-table th {
        border: 1px solid #ddd;
        padding: 8px; /* Bordure grise */
    }

    /* Alternance des couleurs des lignes */
    .param-table tr:nth-child(even){
        background-color: #f2f2f2; /* Gris très clair */
    }

    /* Effet hover sur les lignes */
    .param-table tr:hover {
        background-color: #ddd; /* Gris clair */
    }

    /* Style des en-têtes de tableau */
    .param-table th {
        padding-top: 12px;
        padding-bottom: 12px;
        text-align: left;
        background-color: #2b5876; /* Bleu foncé */
        color: white;
    }

    /* Style des boutons de téléchargement */
    .download-btn {
        background-color: #2b5876 !important; /* Bleu foncé */
        color: white !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        border-radius: 8px !important;
    }

    /* Style des tableaux de diagnostic */
    .diagnosis-table {
        font-size: 0.9rem;
    }

    .diagnosis-table th {
        background-color: #2b5876 !important; /* Bleu foncé */
        color: white !important;
    }

    /* Style du diagnostic actuel */
    .current-diagnosis {
        font-weight: bold;
        border-left: 3px solid #2b5876 !important; /* Bleu foncé */
    }

    /* Légende colorée */
    .legend {
        display: flex;
        gap: 15px;
        margin: 1rem 0;
    }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .legend-color {
        width: 15px;
        height: 15px;
        border-radius: 3px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    models = {}
    try:
        from Models.unet_model import UNet
        seg_model = UNet()
        seg_model.load_state_dict(torch.load('C:/Users/Hocine/Desktop/PFE/coding/diagnostic/public/Models/aml_unet.pth', map_location='cpu'))
        seg_model.eval()
        models['seg_model'] = seg_model

        from Models.vit_model import VisionTransformer
        vit_model = VisionTransformer()
        vit_model.load_state_dict(torch.load('C:/Users/Hocine/Desktop/PFE/coding/diagnostic/public/Models/vit_cell_classifier.pth', map_location='cpu'))
        vit_model.eval()
        models['classif_model'] = vit_model

        models['blood_model'] = joblib.load('C:/Users/Hocine/Desktop/PFE/coding/diagnostic/public/Models/blood_cancer_classifier.pkl')
        models['label_encoder'] = joblib.load('C:/Users/Hocine/Desktop/PFE/coding/diagnostic/public/Models/label_encoder.pkl')

        return models

    except Exception as e:
        raise RuntimeError(f"Erreur de chargement: {str(e)}")

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

for i in range(1, 4):
    status_text.text(f"Étape {i}/3: Chargement des composants...")
    progress_bar.progress(i * 33)
    time.sleep(0.5)

try:
    with st.spinner("Chargement des modèles en cours..."):
        models = load_models()
        st.toast("✅ Modèles chargés avec succès", icon="✅")
        progress_bar.progress(100)
        status_text.success("Prêt !")
except Exception as e:
    st.error(f"""
    **Erreur critique**  
    Impossible de charger les modèles:  
    `{str(e)}`  
    Vérifiez que:  
    - Les fichiers .pth/.pkl existent dans le dossier Models/  
    - Les architectures des modèles correspondent
    """)
    st.stop()

def process_image_segmentation(model, image):
    try:
        import numpy as np
        assert hasattr(np, 'ndarray'), "NumPy mal installé"

        if not np.__version__:
            raise RuntimeError("NumPy non fonctionnel")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])

        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            mask = model(img_tensor).squeeze().cpu().numpy()

        mask = (mask > 0.5).astype(np.uint8)
        return mask

    except ImportError:
        raise RuntimeError("NumPy requis: pip install numpy")
    except Exception as e:
        raise RuntimeError(f"Erreur segmentation: {str(e)}")


def process_image_classification(model, image):
    try:
        # Transformation spécifique au ViT
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])

        img_tensor = transform(image).unsqueeze(0)

        # Prédiction
        with torch.no_grad():
            outputs = model(img_tensor)
            if outputs.shape[1] == 1:
                prob_malin = torch.sigmoid(outputs).item()
                probs = np.array([1 - prob_malin, prob_malin])
            else:
                probs = torch.nn.functional.softmax(outputs, dim=1)[0].numpy()

        print(f"Type de sortie: {type(outputs)}")
        print(f"Shape: {outputs.shape if hasattr(outputs, 'shape') else 'N/A'}")

        return probs

    except Exception as e:
        logger.error(f"Erreur de classification: {str(e)}")
        raise RuntimeError(f"Échec classification: {str(e)}")

def analyze_blood_sample(input_data):
    try:
        data_for_model = {
            'age': input_data['Age'],
            'WBC_G_L': input_data['WBC'],
            'MCV_fL': input_data['MCV'],
            'MCHC_g_L': input_data['MCHC'],
            'ANC_percent': input_data['ANC_percent'],
            'ANC_G_L': input_data['ANC_abs'] / 1000,
            'Lymphocytes_percent': input_data['Lymphocytes_percent'],
            'Lymphocytes_G_L': input_data['Lymphocytes_abs'] / 1000,
            'Monocytes_percent': input_data['Monocytes_percent'],
            'Monocytes_G_L': input_data['Monocytes_abs'] / 1000,
            'Platelets_G_L': input_data['Platelets'],
            'PT_percent': input_data['PT'],
            'Fibrinogen_g_L': input_data['Fibrinogen'],
            'LDH_UI_L': input_data['LDH']
        }

        features_df = pd.DataFrame([data_for_model])

        if debug_mode:
            st.write("Données envoyées au modèle :", features_df)

        with st.spinner("Analyse en cours..."):
            prediction = models['blood_model'].predict(features_df)[0]
            proba = models['blood_model'].predict_proba(features_df)[0]
            diagnosis = models['label_encoder'].inverse_transform([prediction])[0]

        return {
            "diagnosis": diagnosis,
            "confidence": round(max(proba)*100, 2),
            "probas": proba.tolist()
        }

    except Exception as e:
        logger.error(f"Erreur analyse: {str(e)}")
        raise RuntimeError(f"Échec de l'analyse: {str(e)}")

st.markdown('<div class="header">🩸 Assistant de Diagnostic de la LMA</div>', unsafe_allow_html=True)


def get_unit(param_name):
    units = {
        "Age": "ans",
        "WBC": "G/L",
        "MCV": "fL",
        "MCHC": "g/L",
        "ANC_percent": "%",
        "ANC_abs": "G/L",
        "Lymphocytes_percent": "%",
        "Lymphocytes_abs": "G/L",
        "Monocytes_percent": "%",
        "Monocytes_abs": "G/L",
        "Platelets": "G/L",
        "PT": "%",
        "Fibrinogen": "g/L",
        "LDH": "UI/L"
    }
    return units.get(param_name, "")


# Version simplifiée sans PDF
def generate_text_report(results, blood_data):
    report = f"""
    RAPPORT D'ANALYSE HÉMATOLOGIQUE
    ================================

    Diagnostic: {results['diagnosis']}
    Confiance: {results['confidence']}%

    Paramètres Sanguins:
    """
    for param, value in blood_data.items():
        report += f"\n- {param}: {value} {get_unit(param)}"

    return report

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/9/99/AML-M2_-_2.jpg/320px-AML-M2_-_2.jpg",
             caption="Image microscopique de LMA")
    st.markdown("""
    **Instructions:**
    1. Importer une image microscopique
    2. Lancer l'analyse
    3. Consulter les résultats
    """)
    st.markdown("---")
    debug_mode = st.checkbox("Mode debug", False)

tab1, tab2 = st.tabs(["🔬 Diagnostic par Image", "💉 Diagnostic par Analyse Sanguine"])

with tab1:
    st.subheader("Analyse d'Image Microscopique")

    uploaded_file = st.file_uploader("Importer une image",
                                     type=["jpg", "png", "jpeg"],
                                     help="Image microscopique de frottis sanguin")

    if uploaded_file and st.button("Lancer l'Analyse"):
        try:
            image = Image.open(uploaded_file).convert('RGB')

            with st.spinner("Analyse en cours..."):
                mask = process_image_segmentation(models['seg_model'], image)

                class_probs = process_image_classification(models['classif_model'], image)
                diagnosis = "Malin" if class_probs[1] > 0.5 else "Bénin"
                confidence = class_probs[1] if len(class_probs) > 1 else class_probs[0]

                st.success("Analyse terminée!")

                tab_results, tab_details = st.tabs(["📊 Résultats Principaux", "🔍 Détails Techniques"])

                with tab_results:
                    st.markdown("### 📌 Diagnostic Final")

                    if diagnosis == "Malin":
                        diagnostic_emoji = "⚠️"
                        diagnostic_color = "#ff4b4b"
                    else:
                        diagnostic_emoji = "✅"
                        diagnostic_color = "#2ecc71"

                    st.markdown(f"""
                    <div style="background-color:{diagnostic_color}10; 
                                padding:1.5rem; 
                                border-radius:10px;
                                border-left:5px solid {diagnostic_color};
                                margin-bottom:2rem;">
                        <h3 style="color:{diagnostic_color}; margin-top:0;">
                            {diagnostic_emoji} Résultat: {diagnosis} 
                            <span style="font-size:0.8em; color:#555;">(Confiance: {confidence:.1f}%)</span>
                        </h3>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("### 📈 Probabilités de Diagnostic")
                    categories = ['Bénin', 'Malin']
                    fig = go.Figure()
                    fig.add_trace(go.Scatterpolar(
                        r=[class_probs[0], class_probs[1]],
                        theta=categories,
                        fill='toself',
                        name='Probabilité'
                    ))
                    fig.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with tab_details:
                    # Section Technique (existant mais réorganisé)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**Image Originale**")
                        st.image(image, use_container_width=True)
                    with col2:
                        st.markdown("**Segmentation**")
                        fig, ax = plt.subplots()
                        ax.imshow(mask, cmap='viridis')
                        ax.axis('off')
                        st.pyplot(fig)

                    st.markdown("### 🎚️ Contrôles Visuels")
                    opacity = st.slider("Transparence du masque", 0.0, 1.0, 0.5)

                    overlay = Image.fromarray((mask * 255).astype(np.uint8)).convert('RGBA')
                    overlay.putalpha(int(opacity * 255))
                    composite = Image.alpha_composite(
                        image.convert('RGBA'),
                        overlay.resize(image.size)
                    )

                    st.image(composite, caption="Superposition Image/Masque", use_container_width=True)

        except Exception as e:
            st.error(f"Erreur lors de l'analyse: {str(e)}")
            if debug_mode:
                st.exception(e)

with tab2:
    st.subheader("Analyse des Paramètres Sanguins")

    with st.form("blood_form"):
        col2, col1, col3 = st.columns(3)

        with col1:
            st.markdown("**Paramètres de base**")
            age = st.number_input("Âge (années)", min_value=0, max_value=120, value=30)
            wbc = st.number_input("WBC (G/L)", min_value=0.0, max_value=100.0, value=6.5, step=0.1)
            platelets = st.number_input("Plaquettes (G/L)", min_value=0.0, max_value=2000.0, value=250.0)
            fibrinogen = st.number_input("Fibrinogène (g/L)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)

        with col2:
            st.markdown("**Paramètres érythrocytaires**")
            mcv = st.number_input("MCV (fL)", min_value=50.0, max_value=120.0, value=90.0)
            mchc = st.number_input("MCHC (g/L)", min_value=200.0, max_value=400.0, value=330.0)
            pt = st.number_input("Temps de prothrombine (%)", min_value=0, max_value=150, value=100)

        with col3:
            st.markdown("**Paramètres leucocytaires**")
            ldh = st.number_input("LDH (UI/L)", min_value=0, max_value=5000, value=250)
            anc_percent = st.number_input("ANC (%)", min_value=0.0, max_value=100.0, value=50.0)
            anc_abs = st.number_input("ANC (G/L)", min_value=0.0, max_value=50.0, value=3.0)
            lymph_percent = st.number_input("Lymphocytes (%)", min_value=0.0, max_value=100.0, value=30.0)
            lymph_abs = st.number_input("Lymphocytes (G/L)", min_value=0.0, max_value=50.0, value=2.0)
            mono_percent = st.number_input("Monocytes (%)", min_value=0.0, max_value=100.0, value=8.0)
            mono_abs = st.number_input("Monocytes (G/L)", min_value=0.0, max_value=50.0, value=0.5)

        submitted = st.form_submit_button("Analyser")

        if submitted:
            try:
                blood_data = {
                    "Age": age,
                    "WBC": wbc,
                    "MCV": mcv,
                    "MCHC": mchc,
                    "ANC_percent": anc_percent,
                    "ANC_abs": anc_abs,
                    "Lymphocytes_percent": lymph_percent,
                    "Lymphocytes_abs": lymph_abs,
                    "Monocytes_percent": mono_percent,
                    "Monocytes_abs": mono_abs,
                    "Platelets": platelets,
                    "PT": pt,
                    "Fibrinogen": fibrinogen,
                    "LDH": ldh
                }

                with st.spinner("Analyse en cours..."):
                    results = analyze_blood_sample(blood_data)

                st.success("Analyse terminée!")

                DIAGNOSIS_CONFIG = {
                    "ALL": {
                        "color": "#FF6D00",
                        "icon": "🟠",
                        "name": "Leucémie Lymphoïde Aiguë",
                        "key_features": [
                            "Lymphocytes élevés",
                            "Blastes >20%",
                            "CD19+/CD10+",
                            "Atteinte médullaire"
                        ]
                    },
                    "AML": {
                        "color": "#D32F2F",
                        "icon": "🔴",
                        "name": "Leucémie Myéloïde Aiguë",
                        "key_features": [
                            "Myéloblastes",
                            "Auer rods possibles",
                            "CD33+/CD13+",
                            "Hyperleucocytose"
                        ]
                    },
                    "APL": {
                        "color": "#7B1FA2",
                        "icon": "🟣",
                        "name": "Leucémie Promyélocytaire",
                        "key_features": [
                            "DIC fréquente",
                            "Promyélocytes anormaux",
                            "Translocation t(15;17)",
                            "PML-RARA+"
                        ]
                    }
                }

                current_diag = results["diagnosis"]
                diag_info = DIAGNOSIS_CONFIG[current_diag]

                st.success("Analyse hématologique terminée !")

                tab_diagnostic, tab_parameters, tab_biology = st.tabs(
                    ["🩺 Diagnostic", "📊 Paramètres", "🔍 Profil Biologique"])

                with tab_diagnostic:
                    st.markdown(f"""
                        <div style="background:{diag_info['color']}20;
                                    padding:1.5rem;
                                    border-radius:10px;
                                    border-left:5px solid {diag_info['color']};
                                    margin-bottom:2rem;">
                            <div style="display:flex; align-items:center; gap:15px;">
                                <span style="font-size:2.5rem;">{diag_info['icon']}</span>
                                <div>
                                    <h2 style="color:{diag_info['color']}; margin:0;">{diag_info['name']}</h2>
                                    <p style="margin:0; font-size:1.1rem;">Confiance: <strong>{results['confidence']}%</strong></p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("### Probabilités des Diagnostics")
                    fig = go.Figure()

                    for diag in DIAGNOSIS_CONFIG:
                        prob = results["probas"][list(models['label_encoder'].classes_).index(diag)]
                        color = DIAGNOSIS_CONFIG[diag]["color"]
                        fig.add_trace(go.Bar(
                            x=[DIAGNOSIS_CONFIG[diag]["name"]],
                            y=[prob],
                            marker_color=color,
                            text=[f"{prob:.1%}"],
                            textposition='auto'
                        ))

                    fig.update_layout(
                        yaxis=dict(title="Probabilité", range=[0, 1]),
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("### 🎯 Caractéristiques clés")
                    for feature in diag_info["key_features"]:
                        st.markdown(f"- {feature}")

                with tab_parameters:
                    st.markdown("### 📝 Valeurs biologiques")

                    REFERENCE_RANGES = {
                        "WBC": (4.0, 10.0),  # G/L
                        "MCV": (80, 100),  # fL
                        "MCHC": (320, 360),  # g/L
                        "Platelets": (150, 450),  # G/L
                        "ANC_abs": (1.8, 7.0),  # G/L
                        "Lymphocytes_abs": (1.0, 4.0),  # G/L
                        "Monocytes_abs": (0.2, 1.0),  # G/L
                        "Fibrinogen": (2.0, 4.0),  # g/L
                        "LDH": (120, 300)  # UI/L
                    }

                    param_data = []
                    for param, value in blood_data.items():
                        ref_min, ref_max = REFERENCE_RANGES.get(param, (None, None))
                        status = "normal"
                        if ref_min is not None:
                            if value < ref_min:
                                status = "low"
                            elif value > ref_max:
                                status = "high"

                        param_data.append({
                            "Paramètre": param,
                            "Valeur": value,
                            "Unité": get_unit(param),
                            "Référence": f"{ref_min}-{ref_max}" if ref_min else "N/A",
                            "Statut": status
                        })

                    df = pd.DataFrame(param_data)

                    def highlight_status(row):
                        if row["Statut"] == "high":
                            return ["background-color: #ffcccc"] * len(row)
                        elif row["Statut"] == "low":
                            return ["background-color: #ccffff"] * len(row)
                        return [""] * len(row)


                    st.dataframe(
                        df.style.apply(highlight_status, axis=1),
                        hide_index=True,
                        use_container_width=True,
                        column_order=["Paramètre", "Valeur", "Unité", "Référence"]
                    )

                with tab_biology:
                    st.markdown("### Comparaison des profils")

                    PROFILE_DATA = {
                        "Marqueur": ["Blastes", "Lignée", "DIC", "Marqueurs", "Caryotype"],
                        "ALL": [">20%", "Lymphoïde", "Rare", "CD19+/CD10+", "t(12;21)"],
                        "AML": [">20%", "Myéloïde", "Occasionnelle", "CD33+/CD13+", "Variable"],
                        "APL": ["Promyélocytes", "Myéloïde", "Fréquente", "CD33+/CD13+", "t(15;17)"]
                    }

                    st.dataframe(
                        pd.DataFrame(PROFILE_DATA).set_index("Marqueur"),
                        use_container_width=True
                    )

                    report = generate_text_report(results, blood_data)
                    st.download_button(
                        label="📄 Exporter le rapport (TXT)",
                        data=report,
                        file_name=f"rapport_{current_diag}.txt",
                        mime="text/plain"
                    )

            except Exception as e:
                st.error(f"Erreur d'analyse: {str(e)}")
                if debug_mode:
                    st.exception(e)

st.markdown("---")
st.markdown("""
**Système de Diagnostic Assisté par IA**  
© 2023 - Projet Académique
""")