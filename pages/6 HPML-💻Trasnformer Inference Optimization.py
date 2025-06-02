import streamlit as st

st.set_page_config(
    page_icon="🧑‍💻"
)
st.write("""
# HPML-💻Trasnformer Inference Optimization
### Project Links  
💻 **GitHub**: [hpml_proj](https://github.com/weiz-me/hpml_proj)

### Project Description:  
Focused on improving the **inference speed** of popular Transformer models without compromising accuracy. This project combined model fine-tuning, advanced quantization, and memory-efficient attention mechanisms to optimize transformer performance on limited hardware.

### Key Highlights:
- **Model Fine-tuning**:  
  - Applied to **BERT**, **GPT-2**, and **T5** on benchmark datasets (MRPC, COLA)  
  - Achieved **≥85% accuracy** on classification tasks
- **Inference Optimization**:  
  - Applied **post-training dynamic quantization** and **unstructured pruning**  
  - Achieved **2× speedup** during inference with negligible accuracy drop
- **Attention Mechanism Improvements**:  
  - Integrated **Scaled Dot Product Attention (SDPA)** and **Flash Attention 2** into GPT-2  
  - Delivered an additional **1.5× speedup** in inference time with memory efficiency
""")