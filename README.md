# 🧭 Job Description Bias Detector & 📄 OCR Extraction

This project is a **Streamlit dashboard** that combines **Natural Language Processing (NLP)** and **Optical Character Recognition (OCR)** to analyze job descriptions for **bias indicators**.  
It is part of a Final Year Project aligned with **SDG 10: Reduced Inequalities**.

🔗 **Live App**: [job-description-bias-detector.streamlit.app](https://job-description-bias-detector.streamlit.app)

---

## ✨ Features
- **OCR Extractor**  
  Upload job description images (JPG/PNG/TIFF/etc.), extract text using a robust **Tesseract OCR pipeline** (deskew, contrast enhancement, line removal, binarization).

- **Bias Detector (SVM)**  
  Classify job descriptions into bias categories using a trained **TF-IDF + Support Vector Machine** model.

- **Bias Word Dictionary Explorer**  
  Browse/search a curated dictionary of bias terms with categories and severity levels, plus CSV export and bar chart visualizations.

- **Interactive Results**  
  - View preprocessed text used by the model.  
  - Confidence scores and bar chart visualization.  
  - Highlighted text with detected bias terms.  

---

## ⚙️ Tech Stack
- **Python**
- **Streamlit** (UI framework)
- **scikit-learn** (TF-IDF + SVM)
- **spaCy & NLTK** (NLP preprocessing)
- **OpenCV + pytesseract** (OCR pipeline)
- **pandas, numpy** (data handling)

---

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo>.git
   cd <your-repo>
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the app:
   ```bash
   streamlit run Dashboard.py
4. Open the link shown in the terminal (default: http://localhost:8501).


📂 Repository Structure

- Dashboard.py → Main Streamlit dashboard

- SVM/ → Saved pipeline (svm_text_pipeline.pkl)

- Artifacts/ → Label encoder + bias dictionary

- requirements.txt → Dependencies

📌 Notes

- English-only input supported.

- Designed for academic research & demonstration purposes (not production hiring decisions).

- OCR works best on printed, single-page JDs with 150–300 dpi resolution.

📜 License

- This project is for educational and research use.
