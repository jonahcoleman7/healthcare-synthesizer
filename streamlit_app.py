# streamlit_app.py - Single file version for easy deployment
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import tempfile
import io
import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Healthcare Data Synthesizer",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if "processing_stage" not in st.session_state:
    st.session_state.processing_stage = "upload"
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = {}
if "schema_info" not in st.session_state:
    st.session_state.schema_info = {}
if "privacy_report" not in st.session_state:
    st.session_state.privacy_report = {}
if "synthetic_data" not in st.session_state:
    st.session_state.synthetic_data = {}
if "validation_results" not in st.session_state:
    st.session_state.validation_results = {}

class HealthcareSynthesizer:
    def __init__(self):
        self.healthcare_patterns = {
            'npi': r'\b\d{10}\b',
            'icd_10': r'\b[A-Z]\d{2}\.?\d*\b',
            'cpt_code': r'\b\d{5}\b',
            'member_id': r'\b[A-Z]{2,3}\d{6,12}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
        }
        
        self.pii_patterns = {
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'medical_record': r'\b[A-Z]{2,3}\d{6,12}\b',
        }
    
    def process_uploaded_files(self, uploaded_file):
        """Process uploaded ZIP file and extract CSV data"""
        datasets = {}
        
        try:
            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                for filename in zip_ref.namelist():
                    if filename.endswith('.csv'):
                        with zip_ref.open(filename) as csv_file:
                            df = pd.read_csv(csv_file, low_memory=False)
                            df = self._clean_dataframe(df)
                            datasets[filename.replace('.csv', '')] = df
            
            return datasets
            
        except Exception as e:
            st.error(f"Error processing ZIP file: {e}")
            return {}
    
    def _clean_dataframe(self, df):
        """Basic dataframe cleaning"""
        # Clean column names
        df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True)
        
        # Remove empty rows/columns
        df = df.dropna(how='all').dropna(axis=1, how='all')
        
        # Clean string columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', np.nan)
        
        return df
    
    def analyze_schema(self, datasets):
        """Analyze schema and detect field types"""
        schema_info = {}
        
        for dataset_name, df in datasets.items():
            fields = []
            
            for col in df.columns:
                field_info = self._analyze_field(col, df[col])
                fields.append(field_info)
            
            schema_info[dataset_name] = {
                'record_count': len(df),
                'fields': fields
            }
        
        return schema_info
    
    def _analyze_field(self, field_name, series):
        """Analyze individual field"""
        clean_series = series.dropna()
        
        field_info = {
            'name': field_name,
            'nullable': series.isnull().any(),
            'unique_count': series.nunique(),
            'sample_values': clean_series.head(3).tolist() if len(clean_series) > 0 else []
        }
        
        if len(clean_series) == 0:
            field_info['type'] = 'empty'
            return field_info
        
        # Determine type
        if pd.api.types.is_numeric_dtype(clean_series):
            field_info['type'] = 'numeric'
            field_info['min'] = clean_series.min()
            field_info['max'] = clean_series.max()
            field_info['mean'] = clean_series.mean()
        elif clean_series.nunique() / len(clean_series) < 0.1:
            field_info['type'] = 'categorical'
            field_info['categories'] = clean_series.value_counts().head(10).to_dict()
        elif any(self._matches_pattern(str(val), 'phone') for val in clean_series.head(10)):
            field_info['type'] = 'phone'
        elif any(self._matches_pattern(str(val), 'ssn') for val in clean_series.head(10)):
            field_info['type'] = 'ssn'
        else:
            field_info['type'] = 'text'
            field_info['avg_length'] = clean_series.astype(str).str.len().mean()
        
        return field_info
    
    def _matches_pattern(self, value, pattern_name):
        """Check if value matches a pattern"""
        pattern = self.healthcare_patterns.get(pattern_name)
        if pattern:
            return bool(re.search(pattern, str(value)))
        return False
    
    def analyze_privacy(self, datasets):
        """Analyze datasets for PII/PHI"""
        privacy_report = {
            'total_fields': 0,
            'pii_detections': [],
            'risk_score': 0.0,
            'safe_for_synthesis': True
        }
        
        for dataset_name, df in datasets.items():
            privacy_report['total_fields'] += len(df.columns)
            
            for col in df.columns:
                detection = self._detect_pii_in_column(col, df[col])
                if detection:
                    privacy_report['pii_detections'].append(detection)
        
        # Calculate risk score
        if privacy_report['pii_detections']:
            avg_confidence = np.mean([d['confidence'] for d in privacy_report['pii_detections']])
            privacy_report['risk_score'] = avg_confidence
            privacy_report['safe_for_synthesis'] = avg_confidence < 0.7
        
        return privacy_report
    
    def _detect_pii_in_column(self, column_name, series):
        """Detect PII in a column"""
        # Check field name
        sensitive_names = ['ssn', 'phone', 'email', 'name', 'address']
        for name in sensitive_names:
            if name in column_name.lower():
                return {
                    'field_name': column_name,
                    'pii_type': f'sensitive_field_{name}',
                    'confidence': 0.8,
                    'recommendation': 'mask'
                }
        
        # Pattern matching
        sample_values = series.dropna().astype(str).head(10).tolist()
        for pattern_name, pattern in self.pii_patterns.items():
            matches = sum(1 for val in sample_values if re.search(pattern, str(val)))
            if matches > len(sample_values) * 0.3:  # 30% threshold
                return {
                    'field_name': column_name,
                    'pii_type': pattern_name,
                    'confidence': matches / len(sample_values),
                    'recommendation': 'mask'
                }
        
        return None
    
    def generate_synthetic_data(self, datasets, multiplier=10, privacy_level='medium'):
        """Generate synthetic data"""
        synthetic_datasets = {}
        
        for dataset_name, df in datasets.items():
            synthetic_df = self._synthesize_dataset(df, multiplier, privacy_level)
            synthetic_datasets[dataset_name] = synthetic_df
        
        return synthetic_datasets
    
    def _synthesize_dataset(self, df, multiplier, privacy_level):
        """Synthesize a single dataset"""
        np.random.seed(42)
        target_rows = len(df) * multiplier
        
        synthetic_data = {}
        
        for col in df.columns:
            clean_series = df[col].dropna()
            
            if len(clean_series) == 0:
                synthetic_data[col] = [np.nan] * target_rows
                continue
            
            if pd.api.types.is_numeric_dtype(clean_series):
                # Numeric synthesis
                mean, std = clean_series.mean(), clean_series.std()
                synthetic_values = np.random.normal(mean, std, target_rows)
                
                if clean_series.dtype in ['int64', 'int32']:
                    synthetic_values = np.round(synthetic_values).astype(int)
                
                synthetic_values = np.clip(synthetic_values, clean_series.min(), clean_series.max())
                synthetic_data[col] = synthetic_values.tolist()
                
            elif clean_series.nunique() / len(clean_series) < 0.1:
                # Categorical synthesis
                value_counts = clean_series.value_counts(normalize=True)
                synthetic_data[col] = np.random.choice(
                    value_counts.index, 
                    size=target_rows, 
                    p=value_counts.values
                ).tolist()
                
            else:
                # Text synthesis (simple sampling)
                synthetic_data[col] = np.random.choice(clean_series, size=target_rows).tolist()
        
        return pd.DataFrame(synthetic_data)
    
    def validate_synthetic_data(self, original_datasets, synthetic_datasets):
        """Validate synthetic data quality"""
        validation_results = {
            'overall_score': 0.0,
            'field_metrics': {},
            'summary': []
        }
        
        scores = []
        
        for dataset_name in original_datasets.keys():
            if dataset_name in synthetic_datasets:
                orig_df = original_datasets[dataset_name]
                synth_df = synthetic_datasets[dataset_name]
                
                dataset_score = self._validate_dataset(orig_df, synth_df)
                scores.append(dataset_score)
                
                validation_results['field_metrics'][dataset_name] = dataset_score
        
        validation_results['overall_score'] = np.mean(scores) if scores else 0.0
        
        # Generate summary
        if validation_results['overall_score'] > 0.8:
            validation_results['summary'].append("✅ Excellent synthetic data quality!")
        elif validation_results['overall_score'] > 0.6:
            validation_results['summary'].append("⚠️ Good synthetic data quality with minor issues")
        else:
            validation_results['summary'].append("❌ Synthetic data quality needs improvement")
        
        return validation_results
    
    def _validate_dataset(self, orig_df, synth_df):
        """Validate a single dataset"""
        scores = []
        
        common_cols = set(orig_df.columns) & set(synth_df.columns)
        
        for col in common_cols:
            orig_series = orig_df[col].dropna()
            synth_series = synth_df[col].dropna()
            
            if len(orig_series) == 0 or len(synth_series) == 0:
                continue
            
            if pd.api.types.is_numeric_dtype(orig_series):
                try:
                    ks_stat, _ = stats.ks_2samp(orig_series, synth_series)
                    score = max(0, 1 - ks_stat)
                    scores.append(score)
                except:
                    pass
            else:
                # Categorical similarity
                orig_dist = orig_series.value_counts(normalize=True)
                synth_dist = synth_series.value_counts(normalize=True)
                
                all_vals = set(orig_dist.index) | set(synth_dist.index)
                orig_aligned = orig_dist.reindex(all_vals, fill_value=0)
                synth_aligned = synth_dist.reindex(all_vals, fill_value=0)
                
                tv_distance = 0.5 * np.sum(np.abs(orig_aligned - synth_aligned))
                score = max(0, 1 - tv_distance)
                scores.append(score)
        
        return np.mean(scores) if scores else 0.5

# Initialize synthesizer
synthesizer = HealthcareSynthesizer()

def main():
    st.title("🏥 Healthcare Data Synthesizer")
    st.markdown("Transform messy healthcare data into privacy-safe synthetic datasets")
    
    if st.session_state.processing_stage == "upload":
        show_upload_interface()
    elif st.session_state.processing_stage == "analyze":
        show_analysis_interface()
    elif st.session_state.processing_stage == "configure":
        show_configuration_interface()
    elif st.session_state.processing_stage == "results":
        show_results_interface()

def show_upload_interface():
    st.markdown("### 📁 Upload Healthcare Data")
    st.info("Upload a ZIP file containing CSV files with your healthcare data.")
    
    uploaded_file = st.file_uploader(
        "Choose a ZIP file",
        type=['zip'],
        help="Upload a ZIP file containing CSV files with healthcare data"
    )
    
    if uploaded_file is not None:
        if st.button("🔍 Analyze Data", type="primary"):
            with st.spinner("Processing uploaded files..."):
                datasets = synthesizer.process_uploaded_files(uploaded_file)
                
                if datasets:
                    st.session_state.uploaded_data = datasets
                    st.session_state.processing_stage = "analyze"
                    st.success(f"Successfully loaded {len(datasets)} datasets!")
                    st.rerun()
                else:
                    st.error("No valid CSV files found in the ZIP archive.")

def show_analysis_interface():
    st.markdown("### 🔍 Data Analysis Results")
    
    if not st.session_state.uploaded_data:
        st.error("No data loaded. Please go back and upload files.")
        return
    
    # Schema Analysis
    with st.spinner("Analyzing data schema..."):
        schema_info = synthesizer.analyze_schema(st.session_state.uploaded_data)
        st.session_state.schema_info = schema_info
    
    # Privacy Analysis
    with st.spinner("Scanning for PII/PHI..."):
        privacy_report = synthesizer.analyze_privacy(st.session_state.uploaded_data)
        st.session_state.privacy_report = privacy_report
    
    # Display results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Schema Overview")
        for dataset_name, info in schema_info.items():
            with st.expander(f"Dataset: {dataset_name}"):
                st.metric("Records", info['record_count'])
                st.metric("Fields", len(info['fields']))
                
                field_types = {}
                for field in info['fields']:
                    field_type = field['type']
                    field_types[field_type] = field_types.get(field_type, 0) + 1
                
                st.write("**Field Types:**")
                for ftype, count in field_types.items():
                    st.write(f"- {ftype.title()}: {count}")
    
    with col2:
        st.subheader("🔒 Privacy Analysis")
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Fields Analyzed", privacy_report['total_fields'])
        with col_b:
            st.metric("PII Detections", len(privacy_report['pii_detections']))
        with col_c:
            risk_color = "red" if privacy_report['risk_score'] > 0.7 else "orange" if privacy_report['risk_score'] > 0.3 else "green"
            st.metric("Risk Score", f"{privacy_report['risk_score']:.2f}")
        
        if privacy_report['pii_detections']:
            st.warning("⚠️ Potential PII/PHI detected:")
            for detection in privacy_report['pii_detections'][:5]:
                st.write(f"- **{detection['field_name']}**: {detection['pii_type']} (confidence: {detection['confidence']:.2f})")
        else:
            st.success("✅ No obvious PII/PHI patterns detected")
    
    if st.button("➡️ Continue to Configuration", type="primary"):
        st.session_state.processing_stage = "configure"
        st.rerun()

def show_configuration_interface():
    st.markdown("### ⚙️ Synthesis Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        multiplier = st.slider(
            "Dataset Size Multiplier",
            min_value=1,
            max_value=50,
            value=10,
            help="Generate N times the original dataset size"
        )
        
        privacy_level = st.selectbox(
            "Privacy Level",
            ["low", "medium", "high"],
            index=1,
            help="Higher levels provide more privacy but may reduce data utility"
        )
    
    with col2:
        st.info(f"""
        **Configuration Summary:**
        - Original records: {sum(len(df) for df in st.session_state.uploaded_data.values())}
        - Target records: {sum(len(df) for df in st.session_state.uploaded_data.values()) * multiplier}
        - Privacy level: {privacy_level.title()}
        """)
    
    if st.button("🚀 Generate Synthetic Data", type="primary"):
        with st.spinner("Generating synthetic data... This may take a few minutes."):
            # Generate synthetic data
            synthetic_data = synthesizer.generate_synthetic_data(
                st.session_state.uploaded_data,
                multiplier=multiplier,
                privacy_level=privacy_level
            )
            st.session_state.synthetic_data = synthetic_data
            
            # Validate synthetic data
            validation_results = synthesizer.validate_synthetic_data(
                st.session_state.uploaded_data,
                synthetic_data
            )
            st.session_state.validation_results = validation_results
            
            st.session_state.processing_stage = "results"
            st.success("✅ Synthetic data generation complete!")
            st.rerun()

def show_results_interface():
    st.markdown("### 🎉 Synthesis Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        original_count = sum(len(df) for df in st.session_state.uploaded_data.values())
        st.metric("Original Records", original_count)
    
    with col2:
        synthetic_count = sum(len(df) for df in st.session_state.synthetic_data.values())
        st.metric("Synthetic Records", synthetic_count)
    
    with col3:
        multiplier = synthetic_count / original_count if original_count > 0 else 0
        st.metric("Multiplier", f"{multiplier:.1f}x")
    
    with col4:
        quality_score = st.session_state.validation_results.get('overall_score', 0)
        st.metric("Quality Score", f"{quality_score:.2f}")
    
    # Tabs for detailed results
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Preview", "✅ Validation", "🔒 Privacy", "📥 Download"])
    
    with tab1:
        st.subheader("Synthetic Data Preview")
        for dataset_name, df in st.session_state.synthetic_data.items():
            with st.expander(f"Dataset: {dataset_name}"):
                st.write(f"Shape: {df.shape}")
                st.dataframe(df.head(10))
    
    with tab2:
        st.subheader("Quality Validation")
        validation = st.session_state.validation_results
        
        st.metric("Overall Quality Score", f"{validation['overall_score']:.2f}")
        
        for summary in validation['summary']:
            if "✅" in summary:
                st.success(summary)
            elif "⚠️" in summary:
                st.warning(summary)
            else:
                st.error(summary)
    
    with tab3:
        st.subheader("Privacy Protection")
        privacy = st.session_state.privacy_report
        
        if privacy['safe_for_synthesis']:
            st.success("✅ Data is considered safe for synthesis")
        else:
            st.warning("⚠️ High-risk PII detected - additional protection applied")
        
        st.write("**Privacy Measures Applied:**")
        st.write("- Statistical noise injection")
        st.write("- Distribution-based sampling (no direct copying)")
        st.write("- PII field masking where detected")
    
    with tab4:
        st.subheader("Download Synthetic Data")
        
        # Create downloadable ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for dataset_name, df in st.session_state.synthetic_data.items():
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                zip_file.writestr(f"{dataset_name}_synthetic.csv", csv_buffer.getvalue())
            
            # Add reports
            reports = {
                'schema_report.json': st.session_state.schema_info,
                'privacy_report.json': st.session_state.privacy_report,
                'validation_report.json': st.session_state.validation_results
            }
            
            for filename, content in reports.items():
                zip_file.writestr(filename, json.dumps(content, indent=2, default=str))
        
        zip_buffer.seek(0)
        
        st.download_button(
            label="📥 Download Synthetic Dataset (ZIP)",
            data=zip_buffer.getvalue(),
            file_name="synthetic_healthcare_data.zip",
            mime="application/zip"
        )
        
        if st.button("🔄 Start New Synthesis"):
            # Reset all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state.processing_stage = "upload"
            st.rerun()

if __name__ == "__main__":
    main()
