import os
import re
import nltk
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from rake_nltk import Rake
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('punkt_tab')

# Download required NLTK resources
if not os.path.exists(os.path.join(nltk.data.path[0], 'tokenizers/punkt')):
    nltk.download('punkt')
if not os.path.exists(os.path.join(nltk.data.path[0], 'corpora/stopwords')):
    nltk.download('stopwords')

# Load NLP models
biobert_tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
biobert_model = AutoModel.from_pretrained("dmis-lab/biobert-v1.1")

# Transition words and phrases
TRANSITION_WORDS = {
    'addition': ['furthermore', 'additionally', 'moreover', 'also', 'besides'],
    'contrast': ['however', 'nevertheless', 'nonetheless', 'conversely', 'although'],
    'result': ['therefore', 'consequently', 'thus', 'hence', 'as a result'],
    'example': ['for instance', 'specifically', 'particularly', 'notably', 'especially'],
    'summary': ['in conclusion', 'overall', 'in summary', 'to summarize', 'in brief']
}

# Medical domain keywords and terms
MEDICAL_KEYWORDS = [
    # General Medical Terms (100)
    "diagnosis", "treatment", "prognosis", "etiology", "pathology", "symptoms", "signs",
    "disease", "disorder", "syndrome", "condition", "infection", "inflammation",
    "therapy", "medication", "drug", "dosage", "administration", "prescription",
    "clinical", "patient", "hospital", "physician", "surgeon", "nurse",
    "surgery", "procedure", "test", "screening", "imaging", "laboratory",
    "chronic", "acute", "congenital", "acquired", "idiopathic", "iatrogenic",
    "malignant", "benign", "metastasis", "remission", "relapse", "recovery",
    "mortality", "morbidity", "incidence", "prevalence", "epidemiology",
    "antibiotics", "antiviral", "antifungal", "analgesic", "anesthetic",
    "cardiovascular", "respiratory", "gastrointestinal", "neurological", "endocrine",
    "immunological", "dermatological", "hematological", "oncological", "psychiatric",
    "pediatric", "geriatric", "obstetric", "gynecological", "urological",
    "prevention", "prophylaxis", "vaccination", "immunization", "screening",
    "diagnostic", "therapeutic", "palliative", "rehabilitation", "prognostic",
    "pathological", "physiological", "anatomical", "biochemical", "molecular",
    "genetic", "hereditary", "congenital", "developmental", "degenerative",
    "inflammatory", "infectious", "parasitic", "bacterial", "viral",
    "fungal", "protozoal", "helminthic", "prion", "autoimmune",
    "allergic", "hypersensitive", "immunodeficiency", "immunosuppression", "transplant",
    "neoplastic", "carcinogenic", "metastatic", "proliferative", "apoptotic",
    "ischemic", "hypoxic", "anoxic", "thrombotic", "embolic",
    "hemorrhagic", "traumatic", "mechanical", "chemical", "radiation",
    "nutritional", "metabolic", "hormonal", "endocrine", "exocrine",
    "sensory", "motor", "cognitive", "behavioral", "psychological",
    "social", "environmental", "occupational", "epidemiological", "statistical",
    
    # Technical Medical Terms (100)
    "myocardial", "cerebral", "pulmonary", "hepatic", "renal",
    "splenic", "pancreatic", "thyroid", "adrenal", "pituitary",
    "hypothalamic", "parathyroid", "pineal", "thymic", "lymphatic",
    "vascular", "arterial", "venous", "capillary", "lymphatic",
    "neural", "synaptic", "axonal", "dendritic", "myelin",
    "epithelial", "mesenchymal", "connective", "muscular", "skeletal",
    "cartilaginous", "osseous", "tendinous", "ligamentous", "fascial",
    "mucosal", "serosal", "cutaneous", "subcutaneous", "intramuscular",
    "intravenous", "intraarterial", "intrathecal", "intraperitoneal", "intracranial",
    "intraocular", "intraoral", "intranasal", "intratracheal", "intravesical",
    "percutaneous", "transdermal", "transmucosal", "transplacental", "transmembrane",
    "extracellular", "intracellular", "interstitial", "intravascular", "intraluminal",
    "perivascular", "perineural", "peritoneal", "pleural", "pericardial",
    "endothelial", "mesothelial", "epithelial", "glandular", "secretory",
    "absorptive", "excretory", "respiratory", "digestive", "reproductive",
    "endocrine", "exocrine", "paracrine", "autocrine", "juxtacrine",
    "synaptic", "neuromuscular", "neurovascular", "neuroendocrine", "neuroimmune",
    "immunological", "inflammatory", "proliferative", "apoptotic", "necrotic",
    "ischemic", "hypoxic", "anoxic", "thrombotic", "embolic",
    "hemorrhagic", "traumatic", "mechanical", "chemical", "radiation",
    "nutritional", "metabolic", "hormonal", "endocrine", "exocrine",
    "sensory", "motor", "cognitive", "behavioral", "psychological",
    "social", "environmental", "occupational", "epidemiological", "statistical",
    
    # Advanced Medical Terms (100)
    "pathophysiology", "immunopathology", "neuropathology", "histopathology", "cytopathology",
    "molecular", "cellular", "tissue", "organ", "systemic",
    "biochemical", "physiological", "pharmacological", "toxicological", "epidemiological",
    "diagnostic", "therapeutic", "prophylactic", "palliative", "rehabilitative",
    "surgical", "medical", "pharmacological", "radiological", "psychological",
    "preventive", "curative", "supportive", "maintenance", "prophylactic",
    "acute", "subacute", "chronic", "recurrent", "remitting",
    "progressive", "regressive", "stable", "unstable", "critical",
    "mild", "moderate", "severe", "fatal", "terminal",
    "local", "regional", "systemic", "metastatic", "disseminated",
    "primary", "secondary", "tertiary", "quaternary", "quinary",
    "congenital", "acquired", "hereditary", "genetic", "sporadic",
    "infectious", "contagious", "communicable", "transmissible", "zoonotic",
    "inflammatory", "degenerative", "neoplastic", "traumatic", "iatrogenic",
    "autoimmune", "allergic", "hypersensitive", "immunodeficiency", "immunosuppression",
    "metabolic", "endocrine", "nutritional", "toxic", "radiation",
    "mechanical", "chemical", "thermal", "electrical", "acoustic",
    "psychological", "behavioral", "cognitive", "emotional", "social",
    "environmental", "occupational", "recreational", "iatrogenic", "nosocomial",
    
    # Specialized Medical Terms (100)
    "cardiology", "neurology", "oncology", "dermatology", "ophthalmology",
    "otolaryngology", "pulmonology", "gastroenterology", "nephrology", "urology",
    "endocrinology", "rheumatology", "allergy", "immunology", "infectious",
    "pathology", "radiology", "anesthesiology", "psychiatry", "surgery",
    "pediatrics", "geriatrics", "obstetrics", "gynecology", "family",
    "emergency", "critical", "intensive", "preventive", "rehabilitation",
    "occupational", "physical", "speech", "respiratory", "nutrition",
    "pharmacy", "pharmacology", "toxicology", "epidemiology", "biostatistics",
    "genetics", "molecular", "cellular", "tissue", "organ",
    "system", "function", "structure", "development", "aging",
    "injury", "repair", "regeneration", "degeneration", "neoplasia",
    "infection", "inflammation", "immunity", "allergy", "autoimmunity",
    "metabolism", "nutrition", "hormones", "enzymes", "proteins",
    "lipids", "carbohydrates", "nucleic", "acids", "vitamins",
    "minerals", "electrolytes", "fluids", "gases", "hormones",
    "neurotransmitters", "receptors", "signals", "pathways", "networks",
    "systems", "organs", "tissues", "cells", "molecules",
    
    # Additional Medical Terms (100+)
    "prevention", "prophylaxis", "vaccination", "immunization", "screening",
    "diagnostic", "therapeutic", "palliative", "rehabilitation", "prognostic",
    "pathological", "physiological", "anatomical", "biochemical", "molecular",
    "genetic", "hereditary", "congenital", "developmental", "degenerative",
    "inflammatory", "infectious", "parasitic", "bacterial", "viral",
    "fungal", "protozoal", "helminthic", "prion", "autoimmune",
    "allergic", "hypersensitive", "immunodeficiency", "immunosuppression", "transplant",
    "neoplastic", "carcinogenic", "metastatic", "proliferative", "apoptotic",
    "ischemic", "hypoxic", "anoxic", "thrombotic", "embolic",
    "hemorrhagic", "traumatic", "mechanical", "chemical", "radiation",
    "nutritional", "metabolic", "hormonal", "endocrine", "exocrine",
    "sensory", "motor", "cognitive", "behavioral", "psychological",
    "social", "environmental", "occupational", "epidemiological", "statistical",
    "preventive", "curative", "supportive", "maintenance", "prophylactic",
    "acute", "subacute", "chronic", "recurrent", "remitting",
    "progressive", "regressive", "stable", "unstable", "critical",
    "mild", "moderate", "severe", "fatal", "terminal",
    "local", "regional", "systemic", "metastatic", "disseminated",
    "primary", "secondary", "tertiary", "quaternary", "quinary",
    "congenital", "acquired", "hereditary", "genetic", "sporadic",
    "infectious", "contagious", "communicable", "transmissible", "zoonotic",
    "inflammatory", "degenerative", "neoplastic", "traumatic", "iatrogenic",
    "autoimmune", "allergic", "hypersensitive", "immunodeficiency", "immunosuppression",
    "metabolic", "endocrine", "nutritional", "toxic", "radiation",
    "mechanical", "chemical", "thermal", "electrical", "acoustic",
    "psychological", "behavioral", "cognitive", "emotional", "social",
    "environmental", "occupational", "recreational", "iatrogenic", "nosocomial"
]

# Medical abbreviations dictionary
MEDICAL_ABBREVIATIONS = {
    "BP": "blood pressure", "HR": "heart rate", "ECG": "electrocardiogram", "CT": "computed tomography",
    "MRI": "magnetic resonance imaging", "CAD": "coronary artery disease", "COPD": "chronic obstructive pulmonary disease",
    "CHF": "congestive heart failure", "CVD": "cardiovascular disease", "DM": "diabetes mellitus", "HTN": "hypertension",
    "TIA": "transient ischemic attack", "MI": "myocardial infarction", "PE": "pulmonary embolism", "DVT": "deep vein thrombosis",
    "STEMI": "ST elevation myocardial infarction", "NSTEMI": "non-ST elevation myocardial infarction", "ECMO": "extracorporeal membrane oxygenation",
    "ARDS": "acute respiratory distress syndrome", "BLS": "basic life support", "ALS": "advanced life support", "ICU": "intensive care unit",
    "OR": "operating room", "CXR": "chest X-ray", "US": "ultrasound", "CTPA": "CT pulmonary angiogram", "EEG": "electroencephalogram",
    "EMG": "electromyogram", "ECHO": "echocardiogram", "VTE": "venous thromboembolism", "TBI": "traumatic brain injury", "SCI": "spinal cord injury",
    "UTI": "urinary tract infection", "PID": "pelvic inflammatory disease", "BPH": "benign prostatic hyperplasia", "UAP": "unstable angina pectoris",
    "MVA": "motor vehicle accident", "SARS": "severe acute respiratory syndrome", "HIV": "human immunodeficiency virus", "AIDS": "acquired immunodeficiency syndrome",
    "HCV": "hepatitis C virus", "STI": "sexually transmitted infection", "LV": "left ventricle", "RV": "right ventricle", "LVEF": "left ventricular ejection fraction",
    "RVEF": "right ventricular ejection fraction", "ECF": "extracellular fluid", "ICF": "intracellular fluid", "CBC": "complete blood count",
    "CMP": "comprehensive metabolic panel", "BUN": "blood urea nitrogen", "Cr": "creatinine", "GFR": "glomerular filtration rate", "ALT": "alanine aminotransferase",
    "AST": "aspartate aminotransferase", "PT": "prothrombin time", "INR": "international normalized ratio", "PTT": "partial thromboplastin time",
    "D-dimer": "D-dimer test", "C-reactive protein": "C-reactive protein test", "Hb": "hemoglobin", "Hct": "hematocrit", "WBC": "white blood cell",
    "RBC": "red blood cell", "PLT": "platelet", "ABG": "arterial blood gas", "sO2": "oxygen saturation", "PO2": "partial pressure of oxygen",
    "PCO2": "partial pressure of carbon dioxide", "tPA": "tissue plasminogen activator", "LMWH": "low-molecular-weight heparin", "LFT": "liver function tests",
    "CTD": "connective tissue disease", "SLE": "systemic lupus erythematosus", "RA": "rheumatoid arthritis", "OA": "osteoarthritis", "IBD": "inflammatory bowel disease",
    "UC": "ulcerative colitis", "Crohn's": "Crohn's disease", "T1DM": "type 1 diabetes mellitus", "T2DM": "type 2 diabetes mellitus", "IGT": "impaired glucose tolerance",
    "BKA": "below-knee amputation", "AKA": "above-knee amputation", "TKA": "total knee arthroplasty", "THA": "total hip arthroplasty", "BMI": "body mass index",
    "HRCT": "high-resolution CT scan", "PET": "positron emission tomography", "CTC": "circulating tumor cells", "AML": "acute myeloid leukemia", "ALL": "acute lymphoblastic leukemia",
    "CLL": "chronic lymphocytic leukemia", "CML": "chronic myelogenous leukemia", "HCT": "hematopoietic stem cell transplantation", "BMT": "bone marrow transplantation",
    "GSW": "gunshot wound", "SIDS": "sudden infant death syndrome", "NST": "non-stress test", "CTG": "cardiotocography", "IUGR": "intrauterine growth restriction",
    "PPH": "postpartum hemorrhage", "C-section": "cesarean section", "PRBC": "packed red blood cells", "ABO": "blood group", "RH": "Rhesus factor",
    "SCD": "sickle cell disease", "GDM": "gestational diabetes mellitus", "HRT": "hormone replacement therapy", "XR": "X-ray", "D&C": "dilation and curettage",
    "IVF": "in vitro fertilization", "PCOS": "polycystic ovary syndrome", "NPO": "nothing by mouth", "IV": "intravenous", "SC": "subcutaneous", "IM": "intramuscular",
    "TDS": "three times a day", "QID": "four times a day", "BID": "twice a day", "TID": "three times a day", "PRN": "as needed", "PCA": "patient-controlled analgesia",
    "NAC": "N-acetylcysteine", "AKI": "acute kidney injury", "CKD": "chronic kidney disease", "ESRD": "end-stage renal disease"
}

class MedicalTextSummarizer:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.rake = Rake()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.medical_keywords = set(word.lower() for word in MEDICAL_KEYWORDS)
        self.medical_abbreviations = MEDICAL_ABBREVIATIONS
        self.transition_words = TRANSITION_WORDS
        
    def get_biobert_embedding(self, text):
        """Generates BioBERT embeddings for semantic similarity calculations."""
        inputs = biobert_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = biobert_model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        embedding = torch.mean(last_hidden_state, dim=1).squeeze().numpy()
        return embedding

    def preprocess_text(self, text):
        """Clean and normalize text"""
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
        text = re.sub(r'\n+', ' ', text)  # Remove newlines
        
        # Split into sentences and remove duplicates
        sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
        seen = set()
        unique_sentences = [s for s in sentences if not (s.lower() in seen or seen.add(s.lower()))]
        
        return ' '.join(unique_sentences) if unique_sentences else None

    def extract_medical_terms(self, text):
        """Extract medical terms using keyword matching, RAKE, and abbreviations"""
        # Extract medical keywords from text
        words = text.lower().split()
        medical_terms = set()
        
        # Direct keyword matching
        for word in words:
            word = re.sub(r'[^\w\s]', '', word)  # Remove punctuation
            if word in self.medical_keywords:
                medical_terms.add(word)
                
        # Check for medical abbreviations
        for word in text.split():
            if word.isupper() and word in self.medical_abbreviations:
                medical_terms.add(word)
                medical_terms.add(self.medical_abbreviations[word])
        
        # Use RAKE to extract key phrases
        self.rake.extract_keywords_from_text(text)
        rake_phrases = self.rake.get_ranked_phrases()
        
        # Add any RAKE phrase that contains a medical keyword
        for phrase in rake_phrases:
            for word in phrase.lower().split():
                if word in self.medical_keywords:
                    medical_terms.add(phrase)
                    break
        
        # Return as dictionary with simple scores
        return {term: 1.0 for term in medical_terms}

    def has_transition_word(self, sentence):
        """Check if a sentence contains transition words"""
        sentence_lower = sentence.lower()
        for category in self.transition_words.values():
            for word in category:
                if word in sentence_lower:
                    return True
        return False

    def rank_sentences(self, text, medical_terms):
        """Rank sentences based on medical relevance, position, and semantic similarity"""
        sentences = sent_tokenize(text)
        
        # Get sentence embeddings
        sentence_embeddings = [self.get_biobert_embedding(sent) for sent in sentences]
        centroid = np.mean(sentence_embeddings, axis=0)
        
        # Calculate sentence scores
        sentence_scores = []
        seen_content = set()  # Track seen content
        
        for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
            # Check for content similarity using medical terms instead of RAKE
            sentence_terms = set(term.lower() for term in self.extract_medical_terms(sentence))
            content_overlap = len(sentence_terms.intersection(seen_content)) / len(sentence_terms) if sentence_terms else 0
            
            if content_overlap > 0.7:  # Skip if too similar to previous content
                continue
                
            seen_content.update(sentence_terms)
            
            # Position score with modified weighting
            position_score = 1.0 if i == 0 else (0.9 if i < len(sentences) // 4 else 0.7)
            
            # Enhanced medical term scoring
            medical_score = 0
            critical_terms = {
                'diagnosis': 2.5, 'treatment': 2.5, 'prognosis': 2.0,
                'examination': 2.0, 'assessment': 2.0, 'evaluation': 2.0,
                'findings': 2.0, 'results': 2.0, 'revealed': 2.0,
                'demonstrated': 2.0, 'showed': 2.0, 'indicated': 2.0,
                'ecg': 2.5, 'cardiac': 2.5, 'enzymes': 2.0, 'intervention': 2.5,
                'procedure': 2.0, 'medication': 2.0, 'therapy': 2.0
            }
            
            sentence_lower = sentence.lower()
            for term in medical_terms:
                term_lower = term.lower()
                # Higher weight for diagnostic and treatment terms
                term_weight = critical_terms.get(term_lower, 1.0)
                # Additional weight for first mention of a term
                if term_lower not in seen_content:
                    term_weight *= 1.5
                if term_lower in sentence_lower:
                    medical_score += medical_terms[term] * term_weight
            
            # Semantic similarity with context consideration
            semantic_score = cosine_similarity([embedding], [centroid])[0][0]
            
            # Length optimization
            length = len(sentence.split())
            length_score = 1.0 if 10 <= length <= 30 else (0.8 if 30 < length <= 40 else 0.6)
            
            # Context bonus for key medical findings and temporal markers
            context_terms = {
                'revealed': 0.4, 'showed': 0.4, 'demonstrated': 0.4,
                'found': 0.4, 'noted': 0.3, 'observed': 0.3,
                'initially': 0.3, 'subsequently': 0.3, 'following': 0.3,
                'after': 0.3, 'before': 0.3, 'during': 0.3,
                'evaluation': 0.4, 'assessment': 0.4, 'examination': 0.4
            }
            
            context_bonus = sum(score for term, score in context_terms.items() 
                              if term in sentence_lower) / 2  # Normalize multiple matches
            
            # Repetition penalty
            repeated_phrases = sum(1 for phrase in sentence_lower.split() 
                                 if sentence_lower.count(phrase) > 1 and phrase not in self.stop_words)
            repetition_penalty = max(0.5, 1 - (repeated_phrases * 0.1))
            
            # Combine scores with adjusted weights
            final_score = (
                medical_score * 0.40 +      # Medical relevance
                semantic_score * 0.25 +     # Semantic coherence
                position_score * 0.15 +     # Position importance
                length_score * 0.10 +       # Length optimization
                context_bonus * 0.10        # Context importance
            ) * repetition_penalty          # Apply repetition penalty
            
            sentence_scores.append((sentence, final_score))
        
        return sorted(sentence_scores, key=lambda x: x[1], reverse=True)

    def generate_summary(self, text, compression_ratio=0.3):
        """Generate extractive summary using sentence ranking"""
        medical_terms = self.extract_medical_terms(text)
        ranked_sentences = self.rank_sentences(text, medical_terms)
        
        # Calculate target word count based on compression ratio
        total_words = len(text.split())
        target_word_count = max(50, int(total_words * compression_ratio))
        
        # Initialize variables for sentence selection
        selected_sentences = []
        current_word_count = 0
        seen_content = set()
        
        # First, always include the first sentence as it often contains crucial context
        first_sentence = ranked_sentences[0][0]
        selected_sentences.append((first_sentence, ranked_sentences[0][1]))
        current_word_count += len(first_sentence.split())
        
        # Track selected sentence indices to maintain order
        selected_indices = {0}
        
        # Then select sentences based on ranking and compression ratio
        for sentence, score in ranked_sentences[1:]:
            sentence_words = sentence.split()
            word_count = len(sentence_words)
            
            # Skip if adding this sentence would exceed target length
            if current_word_count + word_count > target_word_count * 1.1:  # Allow 10% margin
                continue
                
            # Check for content similarity
            sentence_keywords = set(word.lower() for word in sentence_words 
                                 if word.lower() in self.medical_keywords)
            
            # Calculate overlap with seen content
            overlap_ratio = len(sentence_keywords.intersection(seen_content)) / len(sentence_keywords) if sentence_keywords else 1.0
            
            # More stringent overlap criteria for lower compression ratios
            max_overlap = 0.7 if compression_ratio > 0.3 else 0.5
            
            if overlap_ratio < max_overlap:
                selected_sentences.append((sentence, score))
                seen_content.update(sentence_keywords)
                current_word_count += word_count
                
                if current_word_count >= target_word_count:
                    break
        
        # Sort selected sentences by their original order
        sentence_order = {}
        for i, sentence in enumerate(sent_tokenize(text)):
            sentence_order[sentence] = i
        
        selected_sentences.sort(key=lambda x: sentence_order.get(x[0], 999))
        
        # Join selected sentences with minimal but appropriate transitions
        summary_parts = []
        prev_sentence = None
        
        for i, (sentence, _) in enumerate(selected_sentences):
            current_sentence = sentence
            
            if i > 0:
                # Define semantic relationships between sentences
                prev_lower = prev_sentence.lower()
                curr_lower = current_sentence.lower()
                
                # Only add transitions in specific cases:
                
                # 1. Temporal progression
                if any(marker in curr_lower for marker in {'subsequently', 'following', 'later', 'after'}):
                    # Don't add transition - temporal marker already exists
                    pass
                
                # 2. Clear cause-effect relationship
                elif any(marker in curr_lower for marker in {'therefore', 'resulting in', 'leading to', 'consequently'}):
                    # Don't add transition - causal marker already exists
                    pass
                
                # 3. Diagnostic findings
                elif any(term in curr_lower for term in {'revealed', 'showed', 'demonstrated', 'indicated'}) and \
                     any(term in prev_lower for term in {'examination', 'test', 'imaging', 'laboratory'}):
                    current_sentence = f"This {current_sentence[0].lower()}{current_sentence[1:]}"
                
                # 4. Treatment following diagnosis
                elif any(term in curr_lower for term in {'treated', 'administered', 'given', 'started'}) and \
                     any(term in prev_lower for term in {'diagnosed', 'confirmed', 'identified', 'found'}):
                    current_sentence = f"Based on these findings, {current_sentence[0].lower()}{current_sentence[1:]}"
                
                # 5. Outcome or follow-up
                elif i == len(selected_sentences) - 1 and \
                     any(term in curr_lower for term in {'follow-up', 'outcome', 'discharged', 'recovered'}):
                    current_sentence = f"Subsequently, {current_sentence[0].lower()}{current_sentence[1:]}"
            
            # Ensure proper capitalization of medical terms and abbreviations
            for abbrev in self.medical_abbreviations:
                current_sentence = re.sub(rf'\b{abbrev.lower()}\b', abbrev, current_sentence, flags=re.IGNORECASE)
            
            summary_parts.append(current_sentence)
            prev_sentence = current_sentence
        
        summary = ' '.join(summary_parts)
        highlighted_terms = list(self.extract_medical_terms(summary).keys())
        
        return summary, highlighted_terms
    
    def evaluate_summary(self, original_text, summary):
        """Evaluate summary quality using ROUGE scores"""
        # Calculate ROUGE scores
        rouge_scores = self.rouge_scorer.score(original_text, summary)
        
        # Return scores
        return {
            "rouge1": rouge_scores["rouge1"].fmeasure,
            "rouge2": rouge_scores["rouge2"].fmeasure,
            "rougeL": rouge_scores["rougeL"].fmeasure
        }

# Initialize Flask app
app = Flask(__name__)
summarizer = MedicalTextSummarizer()

@app.route('/')
def index():
    return serve_template()

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        # Get request data
        data = request.get_json()
        text = data.get('text', '')
        compression_ratio = float(data.get('compression_ratio', 0.3))
        
        # Validate input
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Preprocess text
        preprocessed_text = summarizer.preprocess_text(text)
        if not preprocessed_text:
            return jsonify({"error": "Invalid text after preprocessing"}), 400
        
        # Generate summary
        summary, medical_terms = summarizer.generate_summary(
            preprocessed_text, 
            compression_ratio=compression_ratio
        )
        
        # Evaluate summary
        evaluation = summarizer.evaluate_summary(preprocessed_text, summary)
        
        # Prepare response
        response = {
            "summary": summary,
            "medical_terms": medical_terms,
            "original_word_count": len(text.split()),
            "summary_word_count": len(summary.split()),
            "compression_ratio": len(summary.split()) / len(text.split()) if len(text.split()) > 0 else 0,
            "rouge_scores": evaluation
        }
            
        return jsonify(response)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# Improved HTML template with better UI
@app.route('/templates/index.html')
def serve_template():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Medical Text Summarizer</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #f1f1f1;
                background-color: #1e1e2f;
                padding: 20px;
            }

            .container {
                max-width: 1000px;
                margin: 0 auto;
                background-color: #2a2a3d;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
                padding: 30px;
                min-height: 100vh;
            }

            h1 {
                color: #4f46e5;
                margin-bottom: 20px;
                text-align: center;
            }

            .input-section {
                margin-bottom: 30px;
            }

            textarea {
                width: 100%;
                height: 200px;
                padding: 15px;
                font-size: 16px;
                border: 1px solid #555;
                border-radius: 4px;
                margin-bottom: 15px;
                resize: vertical;
                font-family: inherit;
                background-color: #1e1e2f;
                color: #f1f1f1;
            }

            .controls {
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                margin-bottom: 20px;
                gap: 15px;
            }

            .slider-container {
                flex-grow: 1;
                display: flex;
                align-items: center;
                gap: 10px;
                min-width: 250px;
            }

            label {
                font-weight: 600;
                min-width: 150px;
            }

            input[type="range"] {
                flex-grow: 1;
                height: 6px;
                appearance: none;
                background: #444;
                border-radius: 3px;
                outline: none;
            }

            input[type="range"]::-webkit-slider-thumb {
                appearance: none;
                width: 18px;
                height: 18px;
                background: #4f46e5;
                border-radius: 50%;
                cursor: pointer;
            }

            #ratioValue {
                font-weight: 600;
                color: #a5b4fc;
                min-width: 35px;
                text-align: center;
            }

            button {
                background-color: #4f46e5;
                color: white;
                border: none;
                padding: 12px 25px;
                font-size: 16px;
                border-radius: 4px;
                cursor: pointer;
                transition: background-color 0.2s;
                font-weight: 600;
                margin-left: auto;
                display: block;
            }

            button:hover {
                background-color: #4338ca;
            }

            button:disabled {
                background-color: #6b6b9c;
                cursor: not-allowed;
            }

            .loading {
                display: none;
                align-items: center;
                justify-content: center;
                gap: 10px;
                margin-top: 20px;
            }

            .spinner {
                width: 20px;
                height: 20px;
                border: 3px solid #444;
                border-top: 3px solid #4f46e5;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            .result-section {
                display: none;
                margin-top: 30px;
                border-top: 1px solid #444;
                padding-top: 30px;
            }

            h2, h3 {
                color: #a5b4fc;
                margin-bottom: 15px;
            }

            .summary-content {
                background-color: #37374a;
                padding: 20px;
                border-radius: 4px;
                border-left: 4px solid #4f46e5;
                margin-bottom: 25px;
                line-height: 1.8;
            }

            .highlight {
                background-color: #4f46e5;
                color: #fff;
                padding: 2px 4px;
                border-radius: 3px;
                font-weight: 500;
            }

            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
                gap: 15px;
                margin-bottom: 20px;
            }

            .stat-card {
                background-color: #2f2f45;
                padding: 15px;
                border-radius: 4px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.3);
            }

            .stat-title {
                font-size: 14px;
                color: #aaa;
                margin-bottom: 5px;
            }

            .stat-value {
                font-size: 20px;
                font-weight: 600;
                color: #c7d2fe;
            }

            @media (max-width: 768px) {
                .container {
                    padding: 20px;
                }

                .controls {
                    flex-direction: column;
                    align-items: stretch;
                }

                button {
                    width: 100%;
                    margin-top: 15px;
                }

                .stats-grid {
                    grid-template-columns: 1fr;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Medical Text Summarizer</h1>
            <div class="input-section">
                <textarea id="inputText" placeholder="Paste medical text here..."></textarea>
                <div class="controls">
                    <div class="slider-container">
                        <label for="compressionRatio">Compression Ratio:</label>
                        <input type="range" id="compressionRatio" min="0.1" max="0.5" step="0.05" value="0.3">
                        <span id="ratioValue">0.3</span>
                    </div>
                    <button id="summarizeBtn">Summarize</button>
                </div>
                <div class="loading" id="loadingIndicator">
                    <div class="spinner"></div>
                    <span>Generating summary...</span>
                </div>
            </div>
            <div class="result-section" id="result">
                <h2>Summary</h2>
                <div class="summary-content" id="summary"></div>
                <h3>Statistics</h3>
                <div class="stats-grid" id="stats"></div>
            </div>
        </div>

        <script>
            const inputText = document.getElementById('inputText');
            const compressionRatio = document.getElementById('compressionRatio');
            const ratioValue = document.getElementById('ratioValue');
            const summarizeBtn = document.getElementById('summarizeBtn');
            const loadingIndicator = document.getElementById('loadingIndicator');
            const result = document.getElementById('result');
            const summaryContent = document.getElementById('summary');
            const statsGrid = document.getElementById('stats');

            compressionRatio.addEventListener('input', function() {
                ratioValue.textContent = this.value;
            });

            summarizeBtn.addEventListener('click', function() {
                const text = inputText.value.trim();
                if (!text) {
                    alert('Please enter some text to summarize');
                    return;
                }
                summarizeBtn.disabled = true;
                loadingIndicator.style.display = 'flex';
                result.style.display = 'none';

                fetch('/summarize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        text: text,
                        compression_ratio: parseFloat(compressionRatio.value)
                    }),
                })
                .then(response => response.json())
                .then(data => {
                    summarizeBtn.disabled = false;
                    loadingIndicator.style.display = 'none';
                    if (data.error) {
                        alert('Error: ' + data.error);
                        return;
                    }
                    result.style.display = 'block';
                    let summaryHtml = data.summary;
                    for (const term of data.medical_terms) {
                        const escapedTerm = term.replace(/[.*+?^${}()|[\\]\\\\]/g, '\\\\$&');
                        const regex = new RegExp('\\\\b' + escapedTerm + '\\\\b', 'gi');
                        summaryHtml = summaryHtml.replace(regex, '<span class="highlight">$&</span>');
                    }
                    summaryContent.innerHTML = summaryHtml;

                    statsGrid.innerHTML = `
                        <div class="stat-card"><div class="stat-title">Original Text</div><div class="stat-value">${data.original_word_count} words</div></div>
                        <div class="stat-card"><div class="stat-title">Summary Length</div><div class="stat-value">${data.summary_word_count} words</div></div>
                        <div class="stat-card"><div class="stat-title">Compression</div><div class="stat-value">${Math.round(data.compression_ratio * 100)}%</div></div>
                        <div class="stat-card"><div class="stat-title">ROUGE-1 Score</div><div class="stat-value">${(data.rouge_scores.rouge1 * 100).toFixed(1)}%</div></div>
                        <div class="stat-card"><div class="stat-title">ROUGE-2 Score</div><div class="stat-value">${(data.rouge_scores.rouge2 * 100).toFixed(1)}%</div></div>
                        <div class="stat-card"><div class="stat-title">ROUGE-L Score</div><div class="stat-value">${(data.rouge_scores.rougeL * 100).toFixed(1)}%</div></div>
                        <div class="stat-card"><div class="stat-title">Medical Terms</div><div class="stat-value">${data.medical_terms.length}</div></div>
                    `;
                })
                .catch(error => {
                    summarizeBtn.disabled = false;
                    loadingIndicator.style.display = 'none';
                    alert('Error: ' + error);
                });
            });
        </script>
    </body>
    </html>
    """
    return html


if __name__ == '__main__':
    app.run(debug=True) 