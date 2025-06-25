# STREAMLIT MACHINE LEARNING APPLICATION - 3 MODELS : SVM, DT,and ADABOOST
# app.py
import streamlit as st
import os
import joblib
import numpy as np
import logging
import pandas as pd
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_DIR = "models"
MODEL_FILES = {
    "svm": f"{MODEL_DIR}/svm_pipeline.pkl",
    "decision_tree": f"{MODEL_DIR}/dt_pipeline.pkl",
    "adaboost": f"{MODEL_DIR}/adb_pipeline.pkl",
}
CLASS_NAMES = ["Human", "AI"]

# -----------------------------
# LOGGING SETUP
# -----------------------------
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# -----------------------------
# MODEL UTILITIES
# -----------------------------
@st.cache_resource
def load_all_models():
    models = {}
    # Ensure model directory exists
    if not os.path.isdir(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        st.warning(f"Created '{MODEL_DIR}' directory; please add model files.")

    for key, path in MODEL_FILES.items():
        available = os.path.exists(path)
        models[f"{key}_available"] = available
        if available:
            try:
                models[key] = joblib.load(path)
                logging.info(f"Loaded {key} from {path}")
            except Exception as e:
                logging.error(f"Failed to load {key}: {e}")
                st.error(f"Error loading model '{key}'")
    return models

@st.cache_data(show_spinner=False)
def extract_text_features(text: str) -> dict:
    words = text.split()
    sentences = [s for s in text.split('.') if s.strip()]
    return {
        'word_count': len(words),
        'char_count': len(text),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'sentence_count': len(sentences)
    }

def get_prediction_explanation(text: str, model_key: str, prob: float) -> str:
    feats = extract_text_features(text)
    base = f"Confidence: {prob:.1%} | Words: {feats['word_count']} | Chars: {feats['char_count']}"
    if model_key == 'svm':
        return "SVM uses decision boundaries on TF-IDF features. " + base
    if model_key == 'decision_tree':
        return "Decision Tree applies rule-based splits. " + base
    if model_key == 'adaboost':
        return "AdaBoost ensembles weak learners. " + base
    return base


def make_prediction(text: str, model_key: str, models: dict) -> tuple:
    """
    Returns: prediction_label, probabilities array, explanation string
    """
    # Check availability
    if not models.get(f"{model_key}_available", False):
        return None, None, f"Model '{model_key}' not available"

    # Vectorize via pipeline loaded model
    model = models[model_key]
    try:
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba([text])[0]
        else:
            # fallback for SVM without probabilities
            dec = model.decision_function([text])[0]
            p = 1 / (1 + np.exp(-dec))
            probs = np.array([1-p, p])
        pred_idx = int(np.argmax(probs))
        pred_label = CLASS_NAMES[pred_idx]
        explanation = get_prediction_explanation(text, model_key, probs[pred_idx])
        return pred_label, probs, explanation
    except Exception as e:
        logging.error(f"Prediction error for {model_key}: {e}")
        return None, None, "Error during prediction"


def get_available_models(models: dict) -> list:
    opts = []
    for key in ['svm', 'decision_tree', 'adaboost']:
        if models.get(f"{key}_available", False):
            display = key.replace('_', ' ').title()
            opts.append((key, display))
    return opts

# -----------------------------
# UI HELPERS - Enhanced for aesthetics
# -----------------------------
def init_css():
    st.markdown("""
    <style>
        /* Import a cute, professional Google Font */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

        html, body, [class*="st-"] {
            font-family: 'Poppins', sans-serif;
        }

        /* Main Header Styling */
        .main-header {
            font-size: 3rem; /* Larger font size */
            text-align: center;
            margin-bottom: 2rem; /* More space below */
            color: #DA70D6; /* Orchid color */
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1); /* Subtle shadow */
        }

        /* Custom prediction result boxes */
        .ai-pred {
            background-color: #FECACA; /* Light Red */
            color: #C0392B; /* Darker Red for text */
            padding: 1.5rem; /* More padding */
            border-radius: 1.5rem; /* More rounded corners */
            margin-bottom: 1rem;
            text-align: center;
            font-size: 1.5rem;
            font-weight: 600;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* Soft shadow */
        }

        .human-pred {
            background-color: #A7F3D0; /* Light Green */
            color: #27AE60; /* Darker Green for text */
            padding: 1.5rem;
            border-radius: 1.5rem;
            margin-bottom: 1rem;
            text-align: center;
            font-size: 1.5rem;
            font-weight: 600;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }

        /* Sidebar navigation improvements */
        .st-emotion-cache-1cypq8u { /* Target the menu container */
            background-color: #FFFFFF; /* White sidebar */
            border-right: 1px solid #F0F0F0; /* Subtle border */
        }
        
        /* Make the option_menu text and icon colors consistent with the theme */
        .st-emotion-cache-1v0mb7d { /* This targets the option_menu title/selected item */
            color: #4B0082; /* Indigo text */
        }
        .st-emotion-cache-1v0mb7d > div > div { /* This targets the icon in option_menu */
            color: #DA70D6; /* Orchid icon */
        }

        /* Buttons styling */
        .st-emotion-cache-lbt081 { /* Target Streamlit buttons */
            background-color: #FF69B4; /* Hot Pink */
            color: white;
            border-radius: 0.8rem;
            border: none;
            padding: 0.8rem 1.5rem;
            font-size: 1.1rem;
            font-weight: 600;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transition: all 0.3s ease;
        }
        .st-emotion-cache-lbt081:hover {
            background-color: #DA70D6; /* Orchid on hover */
            transform: translateY(-2px); /* Slight lift effect */
            box-shadow: 0 6px 12px rgba(0,0,0,0.25);
        }

        /* Text area styling */
        .st-emotion-cache-1xw8czs textarea { /* Target text area */
            border-radius: 0.8rem;
            border: 1px solid #DA70D6; /* Orchid border */
            padding: 1rem;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.1); /* Inner shadow */
        }

        /* Metric styling */
        .st-emotion-cache-1v0mb7d [data-testid="stMetric"] {
            background-color: #FFFFFF;
            padding: 1rem;
            border-radius: 0.8rem;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            border: 1px solid #F0F0F0;
        }
        .st-emotion-cache-1v0mb7d [data-testid="stMetricValue"] {
            color: #DA70D6; /* Orchid color for metric values */
        }

        /* File uploader styling */
        .st-emotion-cache-1c5c56t { /* Target file uploader input */
            border-radius: 0.8rem;
            border: 1px dashed #DA70D6; /* Dashed orchid border */
            padding: 1.5rem;
            background-color: #FFF0F5; /* Very light pink background */
        }

        /* Dataframe styling - limited direct control, but overall theme helps */
        .st-emotion-cache-snw8fm { /* Targets dataframe */
            border-radius: 0.8rem;
            overflow: hidden; /* Ensures rounded corners apply to content */
        }

        /* Progress bar styling */
        .st-emotion-cache-1v0mb7d [data-testid="stProgress"] > div > div {
            background-color: #FF69B4; /* Hot pink progress bar */
        }

        /* Small headers like "Single Text Analysis" */
        .st-emotion-cache-eczf16 { /* This targets h2 headers */
            color: #DA70D6; /* Orchid color for headers */
            font-weight: 600;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
        }

    </style>
    """, unsafe_allow_html=True)


def menu():
    return option_menu(
        menu_title="Welcome to the Streamlit Human Vs AI Detector!", # More welcoming title
        options=["Home","Single Detection","Batch Processing","Model Comparison","Evaluation","Live Demo","Help"],
        icons=[
            "house-fill",           # Home - filled house
            "magic",                # Single Detection - a magic wand for analysis
            "files",                # Batch Processing - multiple files
            "bar-chart-fill",       # Model Comparison - filled bar chart
            "clipboard2-check-fill",# Evaluation - clipboard with a checkmark
            "play-circle-fill",     # Live Demo - play button
            "question-circle-fill"  # Help - filled question circle
        ],
        menu_icon="robot", # Changed to robot, as it's an AI detector
        default_index=0, 
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "#FFFFFF"}, # White sidebar background
            "icon": {"color": "#DA70D6", "font-size": "20px"}, # Orchid icons
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#FFF0F5"}, # Light pink hover
            "nav-link-selected": {"background-color": "#FFB6C1", "color": "#4B0082"}, # Soft pink selected, Indigo text
        }
    )

# -----------------------------
# MAIN APPLICATION
# -----------------------------
def main():
    st.set_page_config(
        page_title="🌸 AI vs Human Detector ✨", # Cute title with emojis
        page_icon="💖", # Heart emoji for page icon
        layout="wide"
    )
    init_css()
    
    selection = menu()
    models = load_all_models()
    if models is None:
        st.error("Failed to load models. Please ensure models are in the 'models/' folder.")
        return

    # Home Page
    if selection == "Home":
        st.markdown('<h1 class="main-header">🌸 AI vs Human Detector ✨</h1>', unsafe_allow_html=True)
        st.markdown(
            """
            <p style="text-align: center; font-size: 1.2rem; color: #6A5ACD;">
            Unraveling the mystery of text origins, one word at a time!
            </p>
            <p style="text-align: center; font-size: 1rem; color: #4B0082;">
            Navigate through our charming sections using the sidebar to explore AI vs Human text detection.
            </p>
            """, unsafe_allow_html=True
        )
        st.markdown("---") # Visual separator
        
        av = get_available_models(models)
        st.info(f"💖 **Model Status:** {len(av)} out of 3 models are ready to charm you! Currently loaded: **{[name for _,name in av]}**")
        st.markdown(
            """
            <p style="font-size: 0.9rem; color: #6A5ACD;">
            If you're missing models, please ensure your trained `svm_pipeline.pkl`, `dt_pipeline.pkl`, and `adb_pipeline.pkl` files are nestled in the `models/` directory.
            </p>
            """, unsafe_allow_html=True
        )


    # Single Detection Page
    elif selection == "Single Detection":
        st.header("✨ Single Text Analysis ✨")
        st.markdown(
            """
            <p style="font-size: 0.95rem; color: #6A5ACD;">
            Enter any text below and pick your favorite model to see if it's got that human touch or an AI sparkle!
            </p>
            """, unsafe_allow_html=True
        )
        av = get_available_models(models)
        
        # Add a check for available models before selectbox
        if not av:
            st.warning("No models are available for prediction. Please load models first.")
            return

        model_key, model_name = st.selectbox("💖 Choose your Detector Model:", av)
        user_text = st.text_area("✍️ Type or paste your text here to analyze its magic:", height=200, 
                                 placeholder="e.g., 'The quick brown fox jumps over the lazy dog.' or 'As an AI language model, I can...'")
        
        st.markdown("<br>", unsafe_allow_html=True) # Add some space

        if st.button("🚀 Analyze Text"):
            if user_text:
                with st.spinner("💫 Analyzing... this won't take a sec!"):
                    pred, probs, expl = make_prediction(user_text, model_key, models)
                
                if pred:
                    cls = "ai-pred" if pred=="AI" else "human-pred"
                    st.markdown(f"<div class='{cls}'><strong>Prediction: {pred}</strong> (Confidence: {max(probs):.1%})</div>", unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True) # Add space

                    col1, col2 = st.columns(2)
                    col1.metric("👩‍💻 Human Likelihood", f"{probs[0]:.1%}")
                    col2.metric("🤖 AI Likelihood", f"{probs[1]:.1%}")
                    
                    st.markdown("<br>", unsafe_allow_html=True) # Add space
                    st.info(f"🧐 **Explanation:** {expl}")
                else:
                    st.error("Oopsie! Something went wrong during prediction. Please try again.")
            else:
                st.warning("Please enter some text to analyze!")

    # Batch Processing Page
    elif selection == "Batch Processing":
        st.header("✨ Batch Text Analysis ✨")
        st.markdown(
            """
            <p style="font-size: 0.95rem; color: #6A5ACD;">
            Got a whole collection of texts? Upload a `.txt` (one text per line) or `.csv` (first column for texts) file and let our detector magic happen!
            </p>
            """, unsafe_allow_html=True
        )

        uploaded = st.file_uploader("📂 Upload your file here:", type=["txt", "csv"])
        av = get_available_models(models)

        if not av:
            st.warning("No models are available for prediction. Please load models first.")
            return

        model_key, _ = st.selectbox("💖 Select the Model for Batch Processing:", av)

        st.markdown("<br>", unsafe_allow_html=True) # Add some space

        if st.button("🌟 Process File") and uploaded:
            if uploaded.type == "text/plain":
                texts = uploaded.getvalue().decode('utf-8').splitlines()
            else:
                df = pd.read_csv(uploaded)
                texts = df.iloc[:,0].astype(str).tolist()
            
            if not texts:
                st.warning("The uploaded file appears to be empty or in an unsupported format.")
                return

            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, txt in enumerate(texts):
                pred, probs, _ = make_prediction(txt, model_key, models)
                results.append({"Text": txt, "Prediction": pred,
                                "Human Likelihood": f"{probs[0]:.1%}", "AI Likelihood": f"{probs[1]:.1%}"})
                progress_bar.progress((i + 1) / len(texts))
                status_text.text(f"Processing text {i+1}/{len(texts)}...")
            
            status_text.success("🎉 Batch processing complete!")
            res_df = pd.DataFrame(results)
            st.dataframe(res_df, use_container_width=True) # Use container width for better display
            
            st.markdown("<br>", unsafe_allow_html=True) # Add space
            st.download_button(
                label="📥 Download Results as CSV", 
                data=res_df.to_csv(index=False).encode('utf-8'), 
                file_name="ai_human_detection_results.csv", 
                mime="text/csv",
                key="download_csv_button" # Unique key for the button
            )

    # Model Comparison Page
    elif selection == "Model Comparison":
        st.header("✨ Compare Models ✨")
        st.markdown(
            """
            <p style="font-size: 0.95rem; color: #6A5ACD;">
            Curious how our models stack up against each other? Type in some text and watch them compare their insights!
            </p>
            """, unsafe_allow_html=True
        )
        text = st.text_area("📝 Enter text to see how different models analyze it:", 
                            height=150, placeholder="e.g., 'This is a sample text for comparison.'")
        av = get_available_models(models)

        if not av:
            st.warning("No models are available for comparison. Please load models first.")
            return

        st.markdown("<br>", unsafe_allow_html=True) # Add some space

        if st.button("🌈 Compare Models") and text:
            comp = []
            for key,name in av:
                pred, probs, _ = make_prediction(text, key, models)
                # Ensure probs are not None if model fails
                human_prob = probs[0] if probs is not None else 0
                ai_prob = probs[1] if probs is not None else 0
                comp.append({"Model": name, "Prediction": pred,
                             "Human": human_prob, "AI": ai_prob})
            df = pd.DataFrame(comp)
            
            st.subheader("📊 Model Predictions Summary")
            st.table(df[['Model','Prediction']].style.set_properties(**{'background-color': '#FFF0F5', 'color': '#4B0082', 'border': '1px solid #DA70D6'})) # Basic styling for table

            st.subheader("📈 Probability Breakdown by Model")
            fig = go.Figure(data=[
                go.Bar(name='Human Likelihood', x=df['Model'], y=df['Human'], marker_color='#A7F3D0'), # Light Green
                go.Bar(name='AI Likelihood', x=df['Model'], y=df['AI'], marker_color='#FECACA') # Light Red
            ])
            fig.update_layout(
                barmode='group',
                title_text='Human vs. AI Probability by Model',
                font=dict(family="Poppins, sans-serif", size=12, color="#4B0082"), # Poppins font
                xaxis_title="Model",
                yaxis_title="Probability",
                plot_bgcolor='#F8F8FF', # Background color for plot area
                paper_bgcolor='#F8F8FF', # Background color for the entire figure
                margin=dict(l=40, r=40, t=60, b=40),
                legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)', bordercolor='#DA70D6', borderwidth=1),
                hoverlabel=dict(bgcolor="white", font_size=12, font_family="Poppins, sans-serif")
            )
            st.plotly_chart(fig, use_container_width=True) # Make plot responsive

    # Evaluation Page
    elif selection == "Evaluation":
        st.header("✨ Model Evaluation ✨")
        st.markdown(
            """
            <p style="font-size: 0.95rem; color: #6A5ACD;">
            Ready to dive into the nitty-gritty? Upload a CSV file with 'text' and 'label' columns to see how well our SVM model performs!
            </p>
            """, unsafe_allow_html=True
        )
        file = st.file_uploader("📊 Upload CSV for Evaluation:", type=["csv"])
        
        av = get_available_models(models)
        if 'svm' not in [key for key, _ in av]:
            st.warning("SVM model is not available for evaluation. Please ensure 'svm_pipeline.pkl' is in the 'models/' folder.")
            return

        st.markdown("<br>", unsafe_allow_html=True) # Add some space

        if st.button("📈 Run Evaluation") and file:
            with st.spinner('Counting stars... I mean, evaluating!'):
                try:
                    df = pd.read_csv(file)
                    if 'text' not in df.columns or 'label' not in df.columns:
                        st.error("CSV must contain 'text' and 'label' columns.")
                        return
                    
                    y_true = df['label'].astype(str).tolist() # Ensure labels are strings for consistent comparison
                    y_pred = []
                    
                    # Using a progress bar for evaluation
                    eval_progress = st.progress(0)
                    eval_status = st.empty()

                    for i, txt in enumerate(df['text']):
                        pred, _, _ = make_prediction(txt, 'svm', models) # Assuming SVM is the model for evaluation
                        y_pred.append(pred)
                        eval_progress.progress((i + 1) / len(df))
                        eval_status.text(f"Evaluating text {i+1}/{len(df)}...")

                    eval_status.success("💖 Evaluation complete!")

                    st.subheader("🌟 Evaluation Results (SVM Model)")
                    acc = accuracy_score(y_true, y_pred)
                    st.metric("🎯 Accuracy Score", f"{acc:.2%}", delta=None) # Display accuracy as a metric

                    st.subheader("Confusion Matrix")
                    st.json(confusion_matrix(y_true, y_pred).tolist()) # Display as JSON for better readability

                    st.subheader("Classification Report")
                    st.text(classification_report(y_true, y_pred))
                except Exception as e:
                    st.error(f"Oh no! An error occurred during evaluation: {e}")
                    st.info("Please make sure your CSV is correctly formatted with 'text' and 'label' columns.")

    # Live Demo Page
    elif selection == "Live Demo":
        st.header("✨ Live Demo - Do you believe in Magic? ✨")
        st.markdown(
            """
            <p style="font-size: 0.95rem; color: #6A5ACD;">
            Watch our detector predict in real-time as you type! Pick a model and start typing away.
            </p>
            """, unsafe_allow_html=True
        )
        av = get_available_models(models)

        if not av:
            st.warning("No models are available for live demo. Please load models first.")
            return

        key, name = st.selectbox("💖 Choose your Live Model:", av)
        text = st.text_input("✍️ Start typing here:", 
                             placeholder="The more you type, the clearer the prediction will be!")
        
        if text:
            pred, probs, _ = make_prediction(text, key, models)
            if pred:
                confidence = probs[1] if pred=='AI' else probs[0]
                st.markdown(f"####💖 Prediction: <span style='color:{'#C0392B' if pred=='AI' else '#27AE60'};'>{pred}</span>", unsafe_allow_html=True)
                st.progress(confidence)
                st.info(f"Confidence: {confidence:.1%}")
            else:
                st.warning("Couldn't make a live prediction. Try typing more!")
        else:
            st.info("Awaiting your magnificent words... Type something!")


    # Help Page
    elif selection == "Help":
        st.header("💖 Help & Instructions - Your Little Guide! 💖")
        st.markdown(
            """
            <p style="font-size: 1.05rem; color: #4B0082;">
            Welcome to the help desk! Here's everything you need to know to make the most of our AI vs Human Detector:
            </p>
            """
            , unsafe_allow_html=True
        )
        st.markdown("---") # Visual separator

        st.subheader("📌 Navigation Tips")
        st.markdown(
            """
            Use the pretty pink sidebar on the left to jump between different functionalities of the app.
            * **Home:** A warm welcome and model status check.
            * **Single Detection:** Analyze one piece of text at a time.
            * **Batch Processing:** Upload a file for analyzing multiple texts.
            * **Model Comparison:** See how different models predict on the same text.
            * **Evaluation:** Assess the performance of our SVM model with your own labeled data.
            * **Live Demo:** Get instant predictions as you type!
            """
        )

        st.subheader("📁 Model Files - Where to Put Them?")
        st.markdown(
            """
            For the app to work its magic, you need to have your trained machine learning models in a folder named `models/`
            at the same level as this `app.py` file. The expected files are:
            * `svm_pipeline.pkl`
            * `dt_pipeline.pkl`
            * `adb_pipeline.pkl`
            """
        )
        st.info("💡 **Pro Tip:** If you see 'model not available' messages, double-check your `models/` folder!")

        st.subheader("Troubleshooting & Fun Facts")
        st.markdown(
            """
            * **What if I don't have all models?** The app will gracefully show you only the models that are available.
            * **Why is my text predicted as 'AI' when it's mine?** Our models are trained on patterns. Sometimes, even human text can share characteristics with AI-generated text, or vice-versa! It's all part of the fascinating world of ML.
            * **Can I suggest new features?** We'd love to hear your ideas! While this demo is set up, your feedback is super valuable.
            """
        )

if __name__ == '__main__':
    main()

