# Medical Text Summarizer

A web-based NLP application for extractive summarization of medical texts using advanced domain-specific techniques and BioBERT embeddings. This tool helps clinicians, researchers, and students quickly generate concise, medically-relevant summaries from lengthy clinical notes, case reports, or research articles.

## Features
- **Extractive summarization** tailored for medical domain
- **BioBERT-based semantic similarity** for sentence ranking
- **Medical keyword and abbreviation recognition** (500+ terms)
- **Customizable compression ratio** (0.1–0.5)
- **ROUGE score evaluation** for summary quality
- **Modern web UI** with medical term highlighting


## Getting Started

### Prerequisites
- Python 3.7+
- pip

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/medical-text-summarizer.git
   cd medical-text-summarizer
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   The first run will automatically download the BioBERT model and required NLTK data.

## Model & Approach
- **Sentence tokenization:** NLTK
- **Medical term extraction:** Keyword matching, abbreviations, and RAKE
- **Sentence ranking:** Combines medical relevance, semantic similarity (BioBERT), position, length, and context
- **Summary generation:** Selects top-ranked sentences, preserves order, and highlights medical terms

## Customization
- Add or modify medical keywords/abbreviations in `app.py`
- Adjust sentence ranking weights for different summarization behavior