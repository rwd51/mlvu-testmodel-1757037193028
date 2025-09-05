import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import pickle
import traceback
from model import MLModel

# Page configuration
st.set_page_config(
    page_title="TestModel",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown('''
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
''', unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🤖 TestModel</h1>', unsafe_allow_html=True)
st.markdown("**Test Deployment**")

# Model info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🔧 Framework", "sklearn")
with col2:
    st.metric("📊 Type", "classification")
with col3:
    st.metric("⚡ Status", "Active")

st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    try:
        model = MLModel()
        # Load model based on detected framework: sklearn
        import os
        
        # Try to load scikit-learn model
        for file_name in ['model.pkl', 'model.joblib']:
            if os.path.exists(file_name):
                import joblib
                self.model = joblib.load(file_name)
                return self.model
        
        # Try pickle as fallback
        model_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
        if model_files:
            import pickle
            with open(model_files[0], 'rb') as f:
                self.model = pickle.load(f)
                return self.model
        
        raise FileNotFoundError('No compatible model file found')

        return model, None
    except Exception as e:
        return None, str(e)

# Initialize model
with st.spinner('🔄 Loading model...'):
    model, load_error = load_model()

if load_error:
    st.error(f'❌ **Model loading failed:** {load_error}')
    st.info('💡 **Tip:** Make sure your model.py file implements the MLModel class with load_model() and predict() methods.')
    st.stop()

st.success('✅ Model loaded successfully!')

# Input section
# Generate input widgets dynamically
st.sidebar.header("🔧 Input Parameters")

# Try to get input schema from model
try:
    if hasattr(model, 'get_input_schema'):
        schema = model.get_input_schema()
        inputs = {}

        for field_name, field_config in schema.items():
            field_type = field_config.get('type', 'number')
            field_label = field_name.replace('_', ' ').title()

            if field_type == 'number':
                min_val = field_config.get('min', 0.0)
                max_val = field_config.get('max', 100.0)
                default_val = field_config.get('default', min_val)

                inputs[field_name] = st.sidebar.slider(
                    field_label,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val)
                )
            elif field_type == 'select':
                options = field_config.get('options', ['Option 1', 'Option 2'])
                inputs[field_name] = st.sidebar.selectbox(field_label, options)
            else:
                inputs[field_name] = st.sidebar.text_input(field_label)
    else:
        # Default input widgets if no schema available
        st.sidebar.warning('⚠️ No input schema found. Using default inputs.')
        inputs = {}
        inputs['feature_1'] = st.sidebar.number_input('Feature 1', value=0.0)
        inputs['feature_2'] = st.sidebar.number_input('Feature 2', value=0.0)
        inputs['feature_3'] = st.sidebar.number_input('Feature 3', value=0.0)

except Exception as e:
    st.sidebar.error(f'Error generating inputs: {str(e)}')
    inputs = {}
    inputs['input_value'] = st.sidebar.number_input('Input Value', value=0.0)


# Prediction section
col1, col2 = st.columns([2, 1])

with col1:
    # Make prediction
    if st.sidebar.button('🚀 Make Prediction', type='primary', use_container_width=True):
        try:
            with st.spinner('Making prediction...'):
                # Convert inputs to DataFrame
                input_df = pd.DataFrame([inputs])
                
                # Make prediction
                prediction = model.predict(input_df)
            
            # Display results
            st.subheader('🎯 Prediction Results')
            
            # Classification results
            st.success(f'**Predicted Class:** {prediction[0]}')
            
            # Try to show prediction probabilities
            try:
                if hasattr(model.model, 'predict_proba'):
                    probabilities = model.model.predict_proba(input_df)[0]
                    
                    if hasattr(model.model, 'classes_'):
                        classes = model.model.classes_
                        prob_df = pd.DataFrame({
                            'Class': classes,
                            'Probability': probabilities
                        }).sort_values('Probability', ascending=False)
                        
                        st.write('**Prediction Probabilities:**')
                        st.dataframe(prob_df, use_container_width=True)
                        
                        # Create probability chart
                        fig = px.bar(prob_df, x='Class', y='Probability', 
                                   title='Prediction Probabilities')
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as prob_error:
                st.info('Prediction probabilities not available')
            
        except Exception as e:
            st.error(f'❌ Prediction failed: {str(e)}')
            with st.expander('🔍 View error details'):
                st.code(str(e))


with col2:
    st.subheader("📋 Input Summary")
    if 'inputs' in locals():
        st.json(inputs)

    st.subheader("ℹ️ Model Info")
    st.markdown(f"**Framework:** sklearn")
    st.markdown(f"**Type:** classification")

    if st.button("📄 Show Model Code"):
        with st.expander("Model Implementation", expanded=True):
            try:
                with open('model.py', 'r') as f:
                    st.code(f.read(), language='python')
            except:
                st.write("model.py not found")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    🚀 <strong>Deployed with MLVU Platform</strong><br>
    Automated ML Model Deployment Made Simple
</div>
""", unsafe_allow_html=True)
