# ai_human_project1
# 🌸 AI vs Human Detector ✨

A Streamlit application that uses three machine learning models (SVM, Decision Tree, and AdaBoost) to classify text as either **Human-written** or **AI-generated**. Complete with pastel-themed UI, emojis, and Lottie animations! 🎀

---

## 📂 Repository Structure

```text
├── app.py               # Main Streamlit application
├── models/              # Place your trained models here
│   ├── svm_pipeline.pkl         # Serialized SVM text-classification pipeline
│   ├── dt_pipeline.pkl          # Serialized Decision Tree pipeline
│   └── adb_pipeline.pkl         # Serialized AdaBoost pipeline
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

---

## 🚀 Quickstart

1. **Clone the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/ai-human-detector.git
   cd ai-human-detector
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\\Scripts\\activate     # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Add your trained model files**

   Copy your `.pkl` pipelines into the `models/` folder:

   * `svm_pipeline.pkl`

   * `dt_pipeline.pkl`

   * `adb_pipeline.pkl`

   > If the folder doesn’t exist, the app will create it for you on first run.

5. **Run the app**

   ```bash
   streamlit run app.py
   ```

   Then open your browser at [http://localhost:8501/](http://localhost:8501/) to explore the cute UI! 🌸

---

## 🎨 Features & UI Highlights

* **Pastel gradient background** and **Pacifico** header font for a girly, professional look.
* **Emoji‑packed sidebar** navigation with 🏠 Home, 🔮 Single Detection, 📂 Batch, ⚖️ Compare, 📊 Evaluation, 🎯 Live Demo, and ❓ Help.
* **Lottie animation** sparkle on the Home page for instant charm.
* **Stylish cards** for prediction outputs with rounded corners, shadows, and orchid/hot‑pink accents.
* **Tooltips** and **fancy buttons** with hover effects.
* **Progress bars** and **metrics** displayed in themed colors.

---

## 🔍 App Sections

1. **Home**

   * Welcome message, Lottie sparkles, and model status overview.

2. **Single Detection**

   * Enter any text, choose a model, and see **✨ Prediction** cards plus confidence metrics.

3. **Batch Processing**

   * Upload `.txt` or `.csv` to process multiple texts at once, with a progress bar and downloadable results.

4. **Model Comparison**

   * Compare all available models side‑by‑side on the same input, with bar charts showing relative probabilities.

5. **Evaluation**

   * Upload a labeled CSV (`text`, `label`) to compute accuracy, confusion matrix, and classification report for SVM.

6. **Live Demo**

   * Real‑time prediction as you type, complete with progress indicator and styled output.

7. **Help**

   * Guide to navigation, file placement, troubleshooting tips, and fun facts about AI vs. human text.

---

## 🛠 Development & Customization

* **Adding models:** Save new `.pkl` pipelines to `models/` and the app will automatically detect them.
* **Styling tweaks:** Edit the `init_css()` block in `app.py` to customize colors, fonts, borders, etc.
* **Animations:** Swap the Lottie URL in `load_lottie()` for your favorite animation.
* **Performance:** Key functions are cached using `@st.cache_resource` and `@st.cache_data`. Use `streamlit run app.py --clear-cache` if you update code.

---

## 📖 Dependencies

Managed in `requirements.txt`. Key libraries include:

* `streamlit`
* `scikit-learn`
* `joblib`
* `numpy`
* `pandas`
* `plotly`
* `streamlit_option_menu`
* `streamlit_lottie`

Install all with:

```bash
pip install -r requirements.txt
```

---


---

*Built with ❤️ by Aurore Munyaneza*
