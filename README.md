# AI-in-ECG-Analytics

Electrocardiogram (ECG) Analytics is a highly critical task for prevention of Cardiovascular
Disease, which turns out to be a common causae of mortability across the globe.

According to the World Health Organization (WHO), cardiovascular disease is the cause of almost 17.9 million
deaths per year. In addition, it can also inflict big economic impact (cost of over 1.1 trillion by 2025).
Hence, extensive work on AI for prediction of irregular heartbeats has been undertaken. This is especially
the case when manual examination of ECG by healthcare professionals can be limited. This calls for need
of AI technology for automatic analysis and detection of heartbeat defects, particularly when large data
of ECG is prevalent these days.

In this work, a Deep Learning model, in which an Attention Block incorporated into 1D-CNN model, is
developed for prediction of the condition of heartbeat. Time-series flow of ECG is given to the model,
which extracts its sequential and global feature to predict the projection of the heartbeat, which 
serves to predict whether its condition is **Normal** or **Abnormal**. 

### 1D-CNN with Attention

<div align="center">
<img src="https://github.com/user-attachments/assets/696d2ec4-3463-497a-afea-4310df27a2ed" width=85% height=85%>
</div><br />

Firstly, the data is preprocessed to prevent the presence of outliers and overfitting. Wavelet Transform Daubechies 4 is
first applied in order to promote robust noise reduction while preserving important signal characteristics.

<div align="center">
<img src="https://github.com/user-attachments/assets/99b80bc8-5c89-4a88-88ed-6161f13f2413" width=85% height=85%>
</div><br />

Afterwards, baseline correction is made to the output to remove its low frequency baseline drift. Finally, Z-score
normalization is applied to the output to remove any possible outlier and prevent disruption in the learning
process of the model. 

