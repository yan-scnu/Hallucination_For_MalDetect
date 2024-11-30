<img src="Hallucination.webp" alt="Theme" style="zoom:25%;" />

# Leveraging GPT-4's Hallucination for API-based Malware Dynamic Detection







## Abstract

Malware detection plays a crucial role in preventing malware intrusions into computer systems. Recent API-based dynamic analysis methods aim to detect obfuscated and packaged malware using Large Language Models. However, these methods neglected the gap between language text and API sequences in representation and pre-training tasks, resulting in compromised performance. To bridge the gap, a mixture corpora dataset containing language texts and API calls is utilized for pre-training, and we leverage GPT-4’s hallucination to enhance and augment this dataset, enabling model training to represent API call semantics from diverse perspectives. 
Besides, we introduce a targeted Fill-in-the-Blank pre-training strategy, allowing our model to learn the association between API calls and the language texts. Experimental results on two benchmark datasets (*i.e*., *Aliyun* and *Catak*) demonstrate that the proposed model improves detection accuracy by 1.57% and 4.56%, respectively, compared to state-of-the-art methods.



## File Description

```model.py``` The code for the main architecture of the model.

```sample_mixture_data.csv``` A partial sample of a mixture dataset containing API sequences and natural text.

```Aliyun_Preprocess.ipynb```Taking the *Aliyun* dataset as an example, code for processing the API sequence dataset.



## Note

we have created a large-scale mixture corpus dataset consisting of approximately 250k+ samples after the submission. If the paper is accepted, we will fully disclose both the code and the corresponding large-scale dataset to ensure transparency and reproducibility of our work.
