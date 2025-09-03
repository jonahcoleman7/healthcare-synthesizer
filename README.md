# healthcare-synthesizer
Hackathon project

# requirements.txt
streamlit==1.28.1
pandas==2.1.3
numpy==1.25.2
scipy==1.11.4
scikit-learn==1.3.2

# .streamlit/config.toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
headless = true
enableCORS = false
port = 8501

# README.md
# 🏥 Healthcare Data Synthesizer

A Streamlit application that transforms messy healthcare data into privacy-safe synthetic datasets.

## Features

- 📁 **Easy Upload**: ZIP file upload with automatic CSV detection
- 🔍 **Smart Analysis**: Schema inference and PII/PHI detection  
- 🔒 **Privacy Protection**: Automatic sanitization of sensitive data
- 🎯 **Synthetic Generation**: Statistical sampling preserving distributions
- ✅ **Quality Validation**: Comprehensive quality metrics and reporting
- 📥 **One-Click Download**: Complete synthetic dataset with reports

## Quick Start

### Local Development
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### Live Demo
Visit: [Your Streamlit Cloud URL]

## Usage

1. **Upload Data**: ZIP file containing healthcare CSV files
2. **Review Analysis**: Schema detection and privacy scanning
3. **Configure**: Set size multiplier and privacy level  
4. **Generate**: Create synthetic dataset
5. **Download**: Get synthetic data + validation reports

## Sample Data Format

```csv
patient_id,diagnosis_code,service_date,amount,provider_npi
P001,M79.3,2023-01-15,250.00,1234567890
P002,E11.9,2023-01-16,180.50,1234567891
```

## Privacy Features

- PII/PHI pattern detection (SSN, phone, email, medical IDs)
- Risk scoring and safety assessment
- Statistical noise injection
- No direct record copying - only distribution sampling

## Validation Metrics

- Distribution similarity (KS tests, Total Variation)
- Category preservation
- Range and constraint validation
- Overall quality scoring

---

Built for the Optura Healthcare Data Hackathon 🏆
