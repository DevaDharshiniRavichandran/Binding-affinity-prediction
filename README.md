<h1 align="center">
    ATM-TCR
</h1>

ATM-TCR demonstrates how a multi-head self-attention based model can be utilized to learn structural information from protein sequences to make binding affinity predictions.

## Modified Model Architecture:
<img src="ATM-TCR-modified-model/data/fig/modified.png" alt="modified model" width="500"/>

### Modifications Made:
1. **Incorporation of Additional Activation Functions:**
   - Added a ReLU activation function after one of the linear layers in the decoder to introduce non-linearity and enhance feature learning.
   - Used a Sigmoid activation function at the final output layer to produce a probability score for binary classification.

2. **Enhanced Data Normalization:**
   - Applied normalization to the concatenated embeddings of TCR and epitope sequences before passing them to the decoder.

3. **Custom Loss Function:**
   - Integrated a weighted binary cross-entropy loss function to address class imbalance issues in the training data.

4. **Extended Performance Metrics:**
   - Added evaluation metrics such as Matthews Correlation Coefficient (MCC) and Cohen’s Kappa to provide a more comprehensive assessment of model performance.


## References
<ul>
<li> Lee, C. H., Kang, H., Sung, G., Choi, Y. S., Kim, D., Kim, H. S., & Kang, C. Y. (2022). ATM-TCR: Robust deep learning-based TCR-pMHC binding affinity prediction for T-cell immunity. Frontiers in Immunology, 13, 893247. https://doi.org/10.3389/fimmu.2022.893247 </li>
<li> Lee, C., et al. ATM-TCR: Deep Learning-Based TCR-Peptide Binding Prediction. GitHub repository, 2023. https://github.com/Lee-CBG/ATM-TCR/tree/main.</li>
<li>Lee, C., et al. catELMo: Contextualized Embeddings for Protein Sequence Analysis. GitHub repository, 2023. https://github.com/Lee-CBG/catELMo/blob/main/bap.py.</li>
</ul>
