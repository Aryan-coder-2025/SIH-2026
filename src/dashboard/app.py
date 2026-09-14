import json
import os
try:
    import streamlit as st
except ImportError:
    st = None

def load_artifacts(filepath: str) -> list:
    """Load the ExperimentArtifact JSON file."""
    if not os.path.exists(filepath):
        return []
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []

def main():
    if st is None:
        print("Streamlit is not installed. Please install it using `pip install streamlit`.")
        return

    st.set_page_config(page_title="SIH26153 Evaluation Dashboard", layout="wide")
    st.title("Network Attack Forecasting Dashboard (SIH26153)")
    
    artifact_path = os.path.join("artifacts", "qa_mock_experiment.json")
    artifacts = load_artifacts(artifact_path)
    
    if not artifacts:
        st.warning(f"No valid artifact found at `{artifact_path}`. Please run the QA pipeline first.")
        st.info("Run `python src/qa_pipeline.py` to generate the mock evaluation artifacts.")
        return

    # Map horizons to their respective artifacts
    horizon_map = {item.get("forecast_horizon", "Unknown"): item for item in artifacts}
    
    # Sidebar selection
    st.sidebar.header("Configuration")
    selected_horizon = st.sidebar.selectbox(
        "Select Forecast Horizon (seconds)",
        options=sorted(horizon_map.keys())
    )
    
    # Display selected artifact
    artifact = horizon_map[selected_horizon]
    metrics = artifact.get("metrics", {})
    
    # 1. Experiment Info
    st.header(f"Model Configuration")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Experiment Name", artifact.get("experiment_name", "N/A"))
    col2.metric("Model", artifact.get("model_name", "N/A"))
    col3.metric("History Length", f"{artifact.get('history_length', 'N/A')} windows")
    col4.metric("Threshold", artifact.get("threshold", "N/A"))
    
    st.markdown("---")
    
    # 2. Key Metrics
    st.header(f"Performance Metrics (+{selected_horizon}s)")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("F1 Score", f"{metrics.get('f1', 0.0):.3f}")
    m_col2.metric("Precision", f"{metrics.get('precision', 0.0):.3f}")
    m_col3.metric("Recall", f"{metrics.get('recall', 0.0):.3f}")
    m_col4.metric("FPR", f"{metrics.get('fpr', 0.0):.3f}")
    
    m2_col1, m2_col2, m2_col3, m2_col4 = st.columns(4)
    m2_col1.metric("Accuracy", f"{metrics.get('accuracy', 0.0):.3f}")
    m2_col2.metric("Specificity", f"{metrics.get('specificity', 0.0):.3f}")
    
    st.markdown("---")
    
    # 3. Confusion Matrix
    st.header("Confusion Matrix")
    cm_col1, cm_col2 = st.columns(2)
    
    with cm_col1:
        st.markdown("**Predicted Positive (Attack)**")
        st.metric("True Positive (TP)", int(metrics.get("cm_tp", 0)))
        st.metric("False Positive (FP)", int(metrics.get("cm_fp", 0)))
        
    with cm_col2:
        st.markdown("**Predicted Negative (Benign)**")
        st.metric("False Negative (FN)", int(metrics.get("cm_fn", 0)))
        st.metric("True Negative (TN)", int(metrics.get("cm_tn", 0)))
        
    st.markdown("---")
    st.caption(f"Artifact generated at: {artifact.get('created_at', 'Unknown')}")

if __name__ == "__main__":
    main()
