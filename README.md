# Local LLM-based Research Companion

## 1. Problem Statement & Overview
My goal is to create a **local research companion LLM** that can answer queries based on a user’s manuscript and a collection of reference documents. Traditional large-scale LLM services are often cloud-based and may pose confidentiality or bandwidth concerns. By storing data locally and embedding reference texts, this project ensures **data privacy** while maintaining the ability to retrieve and synthesize relevant information. 

The system will:
- **Parse and embed PDFs** (e.g., academic papers, reports) so their content can be recalled when a user asks a question.
- Provide context-aware answers to queries by integrating references from the user’s own manuscript and supporting documents.
- **Fine-tune** the base language model on domain-specific data, enhancing its ability to answer specialized questions.

Overall, the project addresses the challenge of allowing researchers to quickly query and reference multiple documents, even in offline or privacy-sensitive scenarios.

---

## 2. Methodology

### 2.1 Overview of Retrieval-Augmented Generation (RAG)
Retrieval-Augmented Generation combines the strength of **neural language models** with the precision of **information retrieval**. The workflow follows several steps to ensure the LLM has direct, context-rich access to relevant sources—rather than relying solely on the model’s built-in knowledge. Below is an in-depth look at how RAG is implemented in this project.

1. **PDF Parsing & Text Chunking**  
   - **Parsing**: Reference PDFs are read using [pdfminer.six] to extract their textual content. This allows the script to handle academic papers, slide decks, or technical documentation in PDF format.  
   - **Chunking**: The extracted text is split into manageable segments (for example, blocks of up to 1,000 characters with a small overlap to preserve context between chunks).  
     - This chunk size is significant: overly large chunks may degrade retrieval accuracy, and overly small chunks may break important contextual links.

2. **Embedding & Indexing with FAISS**  
   - **Sentence Embedding**: Each text chunk is passed through a **SentenceTransformer** to transform it into a numerical vector. This vector captures the semantic meaning of the chunk.  
   - **FAISS Index**: Vectors are stored in a FAISS index, a highly optimized library for *vector similarity search*. This index allows for rapid retrieval of chunks most relevant to a given query.

3. **Query-Time Chunk Retrieval**  
   - **Encoding the Query**: When the user poses a question, that query is embedded using the same SentenceTransformer.  
   - **Searching the Index**: FAISS compares the query vector to all indexed chunk vectors, retrieving the top-k most relevant text segments. This is significantly more accurate than naive keyword searches, as it accounts for semantic matches.

4. **Prompt Construction**  
   - **Combining Manuscript and Retrieved Chunks**: To generate contextually-aware responses, the system merges the user’s manuscript text and the retrieved chunks into a consolidated prompt.  
   - **Objective**: By giving the LLM curated references (both the user’s own draft and relevant external sources), the system aims to ground responses in factual, localized data rather than random guesses or hallucinations.

5. **LLM Generation**  
   - **Model Inference**: A base model (default is GPT-2 in the provided script) then processes this combined prompt and produces a **context-based** answer.  
   - **Local Pipeline**: Because the script loads the model, tokenizers, and index entirely from the local environment, no external calls are needed, preserving privacy and enabling offline usage.

This RAG approach effectively tackles the knowledge gap in base language models, ensuring the user’s project- or domain-specific documents are the foundation for generated answers.

---

### 2.2 Fine-Tuning with LoRA
While RAG ensures that references are up-to-date and contextually relevant, certain use cases demand **more specialized or technical language** than a base model can reliably provide. This is where **Low-Rank Adaptation (LoRA)** offers a fine-tuning solution that is **lighter** and more **resource-efficient** than conventional fine-tuning.

1. **Motivation for Fine-Tuning**  
   - **Domain-Specific Language**: Some topics involve specialized jargon or advanced scientific notation. The base model may not produce coherent or fully accurate outputs without additional training.  
   - **Custom Style & Format**: If repeated question-and-answer patterns are desired, or if there’s a specific “voice” the LLM should adopt, fine-tuning is a straightforward solution.

2. **How LoRA Works**  
   - **Adapter Layers**: Instead of updating the entire large-scale model (which could have hundreds of millions or billions of parameters), LoRA focuses on training small adapter layers in the attention blocks (often only `q_proj` and `v_proj` in modern Transformer architectures).  
   - **Reduced Computation & Memory**: With fewer parameters to learn, LoRA can significantly cut down the time and hardware needed for training.  
   - **Swappable Adapters**: Because these adapters are separate from the main model weights, the base model can be quickly switched between different LoRA adapters for different tasks or domains.

3. **Implementation Steps**  
   - **Data Preparation**: The system reads a CSV file where each row contains (reference name, query, and answer). These pairs become the training set for fine-tuning.  
   - **LoRA Configuration**: The script sets up parameters like `r`, `lora_alpha`, and the target modules (`["q_proj", "v_proj"]` for most Transformers).  
   - **Training Loop**:  
     1. The script tokenizes each prompt-answer pair.  
     2. The LoRA-wrapped base model performs forward and backward passes, but only updates the adapter parameters.  
     3. After as few as **one or two epochs**, the system can produce significantly improved domain-specific responses.  
   - **Saving & Loading Adapters**: Once training completes, the adapter weights are saved separately, allowing the user to load the fine-tuned model without duplicating the entire base model.

Through LoRA fine-tuning, the research companion evolves beyond a generic text-generation system, gaining **enhanced accuracy** for specialized questions or writing tasks in the target research domain.

---

### 2.3 Creating the Fine-Tuning Dataset
To further tailor the model to this specific research context, a hand-curated dataset of 25 question-answer pairs was created using the following process:

1. **Question Bank Generation**  
   - A set of **25 questions** (listed below) was compiled, each targeting a unique aspect of the reference PDF. These questions ranged from summarizing figures and conclusions to deeper queries about limitations, methodology, and ethical considerations.

<small>

| Questions |  |  |  |  |
|-----------|--|--|--|--|
| **Q1** Summarize the figures in this paper. | **Q2** Summarize the methods used. | **Q3** What are the main conclusions of this paper? | **Q4** What are some anticipated questions by reviewers? | **Q5** What are the main novelties and contributions made by this paper? |
| **Q6** What problem does this paper aim to solve? | **Q7** What are the limitations acknowledged by the authors? | **Q8** What related work does the paper compare itself to? | **Q9** How does this work improve upon previous methods? | **Q10** What are the key results presented in the paper? |
| **Q11** What datasets or benchmarks were used in the experiments? | **Q12** How reproducible are the experiments and results? | **Q13** What assumptions does this study make? | **Q14** What is the theoretical foundation of the proposed method? | **Q15** Are the performance improvements statistically significant? |
| **Q16** What are the implications of this research? | **Q17** What real-world applications could benefit from this work? | **Q18** Is the code or data publicly available? | **Q19** What ablation studies were conducted and what did they show? | **Q20** What hyperparameters were used, and how were they chosen? |
| **Q21** Does the paper include error analysis or failure cases? | **Q22** What future directions do the authors suggest? | **Q23** Are there any ethical considerations discussed? | **Q24** How generalizable are the results across domains or datasets? | **Q25** What are the potential risks or drawbacks of this approach? |

</small>

2. **AI-Assisted Draft Answers**  
   - The questions and reference PDF were provided to GPT 4o, which generated initial draft answers.

3. **Manual Review & Curation**  
   - Each AI-generated answer was **manually reviewed** for correctness, clarity, and completeness.
   - Any inaccuracies were corrected, and the language was refined or expanded as needed.
   - The final approved answers were then **paired with the original questions** to form high-quality training examples.

4. **CSV Construction**  
   - Each question–answer pair was added to the CSV file in the format:
     ```
     reference_filename, question, answer
     ```
   - This CSV forms the **fine-tuning dataset** that LoRA uses to further adapt the base model.

With this curated dataset, the model can learn from **high-quality Q&A pairs** specifically tailored to the user’s reference material, thereby improving its domain-specific performance during inference.

---

### 2.4 Pipeline Workflow
<p align="center">
   <img src="research_companion_flowchart.png" alt="workflow" style="width:50%;">
</p>

---

### 2.5 Data Card
Below is an overview of the models used in this project and the relevant licenses or restrictions:

1. **GPT-2**  
   - **Usage**: Used as the base model for inference and demonstration in this project.  
   - **License**: GPT-2’s model code and weights are released by OpenAI under a permissive license (MIT for the code; weights are also permitted for use, but always verify the latest terms from the official source).  
   - **Restrictions & Considerations**: Users should comply with the broader OpenAI usage policies and any local regulations regarding data usage and storage.

2. **GPT 4o**  
   - **Usage**: Employed to generate certain question-answer pairs and as an alternative or more advanced LLM for demonstration or comparison.  
   - **License**: GPT 4o is proprietary software. Usage must adhere to the relevant terms and conditions specified by its providers (which may include usage limits, restrictions on commercial applications, and API-based access if applicable).  
   - **Restrictions & Considerations**: As this model is not fully open-source, distribution or modification of the underlying weights is typically disallowed. This should not be a concern for the use case in this project.

3. **Proprietary Algorithms**  
   - **Status**: No additional proprietary algorithms have been used in this project aside from GPT 4o.  
   - **Note**: All retrieval, embedding, and fine-tuning methods employed here rely on open-source frameworks and libraries (e.g., FAISS, SentenceTransformer, LoRA) which have their own permissive licenses.

---

**In summary, this project’s methodology balances lightweight retrieval-augmented generation with an optional adapter-based fine-tuning step.** By referencing locally stored documents and selectively training adapters on domain-specific data, the user can expect relevant, private, and accurately tailored outputs from their own research companion LLM.

## 3. Results
### 3.1 **Code Walkthrough**

---

### 3.2 **Demo**

---

### 3.3 Planned Quantitative Validation

In this experiment, the performance of three different model setups will be evaluated across five ongoing research projects:

1. **Base Model**  
2. **RAG + Base Model** (i.e., retrieval-augmented generation with the base model)  
3. **RAG + Fine-Tuned Model** (i.e., retrieval-augmented generation with the fine-tuned model)

For each of the five projects:
- Two lab members will each compare the responses to **5 questions**.  
- They will provide pairwise preferences among the different model outputs.  
  1. Base Model vs. RAG + Base Model  
  2. Base Model vs. RAG + Fine-Tuned Model  
  3. RAG + Base Model vs. RAG + Fine-Tuned Model  

From these comparisons, we will compute the **winrate** for:
- **RAG + Base Model** over Base Model  
- **RAG + Fine-Tuned Model** over Base Model  
- **RAG + Fine-Tuned Model** over RAG + Base Model  

The following tables are placeholders for recording and summarizing the eventual winrate data. Fill in the results once you complete the pairwise evaluations.

---

#### RAG + Base Model vs. Base Model

| Project   | Win Rate (RAG + Base) | Comments |
|-----------|------------------------|----------|
| Project 1 |                        |          |
| Project 2 |                        |          |
| Project 3 |                        |          |
| Project 4 |                        |          |
| Project 5 |                        |          |
| **Average** |                      |          |

---

#### RAG + Fine-Tuned Model vs. Base Model

| Project   | Win Rate (RAG + Fine-Tuned) | Comments |
|-----------|-----------------------------|----------|
| Project 1 |                             |          |
| Project 2 |                             |          |
| Project 3 |                             |          |
| Project 4 |                             |          |
| Project 5 |                             |          |
| **Average** |                           |          |

---

#### RAG + Fine-Tuned Model vs. RAG + Base Model

| Project   | Win Rate (RAG + Fine-Tuned) | Comments |
|-----------|-----------------------------|----------|
| Project 1 |                             |          |
| Project 2 |                             |          |
| Project 3 |                             |          |
| Project 4 |                             |          |
| Project 5 |                             |          |
| **Average** |                           |          |

## 4. Critical Analysis & Future Work

- **Assessment & Evaluation**  
  - **Current Implementation’s Failure**  
    The repeated failures of the current retrieval-augmented approach—where outputs are often incorrect or inconsistent—suggest that the model is not effectively integrating the retrieved chunks or aligning its generation with the context provided. Several factors may be contributing to this:
    1. **Insufficient Model Capacity or Alignment**: The chosen base model might lack the necessary parameters or pre-training distribution to handle specialized or technical queries even with retrieval.  
    2. **Ineffective Prompt Construction**: Combining the user’s manuscript and retrieved chunks may be producing prompts that are too lengthy or disjointed, causing confusion in the LLM’s attention mechanism.  

- **Next Steps**  
  - **Improve Retrieval & Prompt Engineering**  
     1. Refine the chunking and embedding methods to ensure better semantic capture.  
     2. Experiment with more advanced prompt templates or short “prompt engineering recipes” to ensure the LLM receives well-structured and relevant context.  
  - **Explore Larger or More Specialized Base Models**  
     1. Switch to a model that has demonstrated strong performance on specialized tasks.  
     2. Consider domain-specific models if the research context is highly technical or niche.  
  - **Refine Fine-Tuning Approach**  
     1. Conduct thorough hyperparameter sweeps to see if the LoRA settings (e.g., learning rate, `r`, etc.) better align the model with the domain.  
     2. Expand the fine-tuning dataset beyond 25 Q&A pairs, or incorporate more diverse question types, to enrich the model’s coverage.  
  - **Systematic Quality Checks**  
     1. Implement intermediate checks or “chain-of-thought” gating to detect nonsense outputs early.  
     2. Integrate a validation step where the system compares each generation against key reference facts before finalizing an answer.

- **Impact & Future Potential**  
  - **Relevant User Groups**  
    Despite the current setbacks, the premise of a locally hosted research companion still holds substantial potential. Once the system can correctly retrieve and contextualize reference materials, its **offline and privacy-preserving** design would benefit:  
    1. **Academic Researchers**: Quick and secure summarization of multiple PDFs without risking data confidentiality.  
    2. **Industry Practitioners**: Handling proprietary or sensitive documents where data cannot be sent to external servers.  
    3. **Educational Environments**: Providing students and teachers with a controlled environment to query course materials and references without relying on cloud services.

  - **Promised Impact if Successful**  
    If these improvements lead to a consistent and reliable workflow, the research companion will:  
    1. **Facilitate Rapid Literature Review**: Researchers can ask complex questions and receive focused, reference-backed answers, reducing time spent manually scanning documents.  
    2. **Maintain Confidentiality**: With all data processed locally, proprietary research can be explored without privacy concerns.  
    3. **Adapt to New Domains**: A robust LoRA fine-tuning method would allow for quick domain shifts, letting the tool serve diverse fields—from biomedical research to legal document analysis.  
    4. **Empower Offline Workflows**: In environments with limited or regulated internet access, a fully offline system still offers high-quality retrieval and generation.

## 5 Documentation & Resources
### 5.1 **Repo Structure & Setup**  
  1. Clone the [LocalResearchLLM repository](https://github.com/MPillaia/LocalResearchLLM)  
     &nbsp;&nbsp;&nbsp;&nbsp;- Note that sample manuscript and references are not provided to prevent improper distribution of published and unpublished work.
  2. Install required dependencies with `pip install -r requirements.txt`.
  3. Create a directory named "references" and include PDF files of your desired references.
  4. Create your fine-tuning CSV file. Any set of question/answer pairs are acceptable, as long as the CSV is in the following format:

     | Reference file name | Question         | Answer         |
     |---------------------|------------------|----------------|
     | reference1.pdf      | Sample Question  | Sample Answer  |

  5. Create a PDF copy of your manuscript.
  6. To complete the RAG processing and fine-tune the base model, run:
     ```bash
     python assistant.py --manuscript_file /path/to/manuscript/my_manuscript.pdf --reference_folder /path/to/references --save_rag --fine_tune --finetune_csv_file /path/to/data/finetuning_samples.csv 
     ```
  7. After completion, to run your research companion without re-processing, run:
     ```bash
     python assistant.py --manuscript_file /path/to/manuscript/my_manuscript.pdf --load_rag --load_model_path /path/to/fine_tuned_model
     ```
     &nbsp;&nbsp;&nbsp;&nbsp;- Note that by default the RAG files and `fine_tuned_model` directory will be stored in your working directory. You can specify the specific RAG file locations with the `--rag_index_file` and `--rag_chunks_file` flags.

---

### 5.2 **Resources & References**  
  - **Resources/References for Described Methodology**

      - **FAISS**  
        - Johnson, J., Douze, M., & Jégou, H. (2017). *Billion-scale similarity search with GPUs.* IEEE Transactions on Big Data, 7(3):535–547.  
        - [Paper](https://arxiv.org/abs/1702.08734)
   
      - **LoRA (Low-Rank Adaptation of Large Language Models)**  
        - Hu, E., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, L., & Chen, W. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*  
        - [Paper](https://arxiv.org/abs/2106.09685)
      
      - **GPT-2**  
        - Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI Blog.  
        - [Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
      
      - **Retrieval-Augmented Generation**  
        - Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-T., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* In *Advances in Neural Information Processing Systems (NeurIPS).*  
        - [Paper](https://arxiv.org/abs/2005.11401)
       
   - **References Used for Demo**
      - Gandelman, J. S., et al. (2019). *The Anatomic Distribution of Skin Involvement in Patients with Incident Chronic Graft-versus-Host Disease.* Biology of Blood and Marrow Transplantation, 25(2), 279.
         - [Paper](https://pubmed.ncbi.nlm.nih.gov/30219700/)
      - Baumrin, E., et al. (2023). *Prognostic Value of Cutaneous Disease Severity Estimates on Survival Outcomes in Patients with Chronic Graft-vs-Host Disease.* JAMA Dermatology, 159(4), 393–402.
         - [Paper](https://pubmed.ncbi.nlm.nih.gov/36884224/)
      - Tkaczyk, E. R., et al. (2018). *Overcoming human disagreement assessing erythematous lesion severity on 3D photos of chronic graft-versus-host disease.* Bone Marrow Transplantation, 53(10), 1356–1358.
         - [Paper](https://pubmed.ncbi.nlm.nih.gov/29740182/)
      - McNeil, A. J., et al. (2024). *Improving AI Assessment of Cutaneous Chronic Graft-Versus-Host Disease using Unlabeled Patient Photographs.* Journal of Clinical and Translational Science, 8(1), 92–92.
         - [Paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11033750/)
      - McNeil, A. J., et al. (2022). *Segmentation of cutaneous chronic graft-versus-host disease by a deep learning neural network.* Journal of Investigative Dermatology, 142(8), 142–142.
         - [Paper](https://pubmed.ncbi.nlm.nih.gov/35322873/)
      - Calin, M. A., et al. (2023). *Mapping the Distribution of Melanin Concentration in Different Fitzpatrick Skin Types Using Hyperspectral Imaging Technique.* Photochemistry and Photobiology, 99(3), 1020–1027.
         - [Paper](https://pubmed.ncbi.nlm.nih.gov/36135823/)
      - Nkengne, A., et al. (2018). *SpectraCam®: A new polarized hyperspectral imaging system for repeatable and reproducible in vivo skin quantification of melanin, total hemoglobin, and oxygen saturation.* Skin Research and Technology, 24(1), 99–107.
         - [Paper](https://pubmed.ncbi.nlm.nih.gov/28771832/)
      - Isensee, F., et al. (2021). *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.* Nature Methods, 18(2), 203–211.
         - [Paper](https://www.nature.com/articles/s41592-020-01008-z)
---
