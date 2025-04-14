# Local LLM-based Research Companion

## 1. Problem Statement & Overview
In this project, the goal is to create a **local research companion LLM** that can answer queries based on a user’s manuscript and a collection of reference documents. Traditional large-scale LLM services are often cloud-based and may pose confidentiality or bandwidth concerns. By storing data locally and embedding reference texts, this project ensures **data privacy** while maintaining the ability to retrieve and synthesize relevant information. 

The system will:
- **Parse and embed PDFs** (e.g., academic papers, reports) so their content can be recalled when a user asks a question.
- Provide context-aware answers to queries by integrating references from the user’s own manuscript and supporting documents.
- **Optionally fine-tune** the base language model on domain-specific data, enhancing its ability to answer specialized questions.

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
   - **Sentence Embedding**: Each text chunk is passed through a **SentenceTransformer** (e.g., "all-MiniLM-L6-v2") to transform it into a numerical vector. This vector captures the semantic meaning of the chunk.  
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
  - Reflect on whether your approach met the project goals.  
  - Discuss strengths and limitations based on data or performance metrics.
- **Impact and Insights**  
  - Consider the broader implications of your solution.  
  - Suggest who benefits most from this project, or where it can be applied.
- **Next Steps**  
  - Propose ways to improve, extend, or refine your work.

## 5. Documentation & Resources
- **Repo Structure & Setup**  
  - Link to repository.  
  - Summarize installation instructions and usage guidelines.  
  - Mention any known issues or troubleshooting tips.
- **Further Reading & References**  
  - Cite key papers, articles, or code bases.  
  - Link to any relevant open-source projects or datasets.

---
**Additional Considerations (Optional or Embedded Where Relevant):**
- **Model & Data Cards**  
  - Outline model version/architecture and intended uses.  
  - Provide licensing information and highlight any ethical/bias considerations.
- **Presentation & Delivery Notes**  
  - Keep explanations concise for a 10–15 minute run-through.  
  - Use visuals or demonstrations sparingly (brief code snippets, images) to support key points.
- **Q&A / Discussion**  
  - Anticipate common questions.  
  - Prepare quick clarifications regarding methodology or results.
