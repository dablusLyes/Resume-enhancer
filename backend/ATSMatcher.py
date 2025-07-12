import re
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np

class ATSMatcher:
    def __init__(self):
        """Initialize the French ATS matcher with required models and data"""
        try:
            # Download required NLTK data
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            
            # Initialize NLP tools for French
            self.stemmer = SnowballStemmer('french')
            self.stop_words = set(stopwords.words('french'))
            
            # Load spaCy model for French NER
            try:
                self.nlp = spacy.load("fr_core_news_sm")
            except:
                print("Warning: spaCy French model not found. Install with: python -m spacy download fr_core_news_sm")
                self.nlp = None
            
            # Initialize sentence transformer for semantic similarity
            try:
                self.sentence_model = SentenceTransformer('distiluse-base-multilingual-cased')
            except:
                print("Warning: SentenceTransformer not available. Install with: pip install sentence-transformers")
                self.sentence_model = None
            
            # French skill categories and keywords
            self.skill_categories = {
                'programmation': ['python', 'java', 'javascript', 'c++', 'c#', 'sql', 'r', 'scala', 'go', 'rust', 'php', 'ruby', 'swift', 'kotlin'],
                'developpement_web': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'développement web'],
                'science_donnees': ['machine learning', 'deep learning', 'tensorflow', 'pytorch', 'pandas', 'numpy', 'scikit-learn', 'keras', 'apprentissage automatique', 'intelligence artificielle', 'ia'],
                'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform', 'ansible', 'infonuagique'],
                'bases_donnees': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle', 'sqlite', 'base de données'],
                'outils': ['git', 'jenkins', 'jira', 'confluence', 'slack', 'trello', 'postman']
            }
            
            # French education keywords
            self.education_keywords = [
                'licence', 'master', 'doctorat', 'diplôme', 'université', 'école', 'certification',
                'informatique', 'ingénierie', 'mathématiques', 'statistiques', 'mba', 'bac+3', 'bac+5',
                'formation', 'études', 'cursus', 'bts', 'dut', 'iut', 'grande école', 'écoles d\'ingénieurs'
            ]
            
            # French experience level keywords
            self.experience_levels = {
                'junior': ['débutant', 'junior', 'diplômé', 'stagiaire', 'apprenti', 'niveau débutant'],
                'intermediaire': ['confirmé', 'expérimenté', 'senior', 'chef', 'responsable', 'niveau intermédiaire'],
                'senior': ['senior', 'principal', 'architecte', 'manager', 'directeur', 'expert', 'lead']
            }
            
        except Exception as e:
            print(f"Erreur lors de l'initialisation de l'ATS Matcher: {e}")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF with better error handling"""
        try:
            text = ""
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text.strip()
        except Exception as e:
            print(f"Erreur lors de l'extraction du texte PDF: {e}")
            return ""
    
    def read_text_file(self, file_path: str) -> str:
        """Read text file with encoding detection"""
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read().strip()
                except UnicodeDecodeError:
                    continue
            return ""
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier texte: {e}")
            return ""
    
    def preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing for French ATS"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\+\#\.]', ' ', text)
        
        # Handle French accents normalization
        text = text.replace('é', 'e').replace('è', 'e').replace('ê', 'e').replace('ë', 'e')
        text = text.replace('à', 'a').replace('â', 'a').replace('ä', 'a')
        text = text.replace('ù', 'u').replace('û', 'u').replace('ü', 'u')
        text = text.replace('ô', 'o').replace('ö', 'o')
        text = text.replace('î', 'i').replace('ï', 'i')
        text = text.replace('ç', 'c')
        
        return text.strip()
    
    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extract skills from French text using keyword matching"""
        text_lower = text.lower()
        found_skills = defaultdict(list)
        
        for category, skills in self.skill_categories.items():
            for skill in skills:
                if skill.lower() in text_lower:
                    found_skills[category].append(skill)
        
        return dict(found_skills)
    
    def extract_education(self, text: str) -> List[str]:
        """Extract education information from French text"""
        text_lower = text.lower()
        found_education = []
        
        for edu_keyword in self.education_keywords:
            if edu_keyword in text_lower:
                found_education.append(edu_keyword)
        
        return found_education
    
    def extract_experience_years(self, text: str) -> Optional[int]:
        """Extract years of experience from French text"""
        patterns = [
            r'(\d+)\s*(?:\+)?\s*an(?:s|née)?s?\s*(?:d[\'e]\s*)?expérience',
            r'(\d+)\s*(?:\+)?\s*années?\s*(?:d[\'e]\s*)?expérience',
            r'expérience\s*(?:de\s*)?(\d+)\s*(?:\+)?\s*an(?:s|née)?s?',
            r'(\d+)\s*(?:\+)?\s*an(?:s|née)?s?\s*dans\s*(?:le\s*)?(?:domaine|secteur)',
            r'(\d+)\s*(?:\+)?\s*an(?:s|née)?s?\s*de\s*(?:pratique|métier)',
            r'plus\s*de\s*(\d+)\s*an(?:s|née)?s?'
        ]
        
        text_lower = text.lower()
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return int(match.group(1))
        
        return None
    
    def calculate_skill_match(self, resume_skills: Dict[str, List[str]], 
                            job_skills: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate skill matching scores by category"""
        skill_scores = {}
        
        for category in set(resume_skills.keys()) | set(job_skills.keys()):
            resume_cat_skills = set(resume_skills.get(category, []))
            job_cat_skills = set(job_skills.get(category, []))
            
            if job_cat_skills:
                intersection = resume_cat_skills.intersection(job_cat_skills)
                score = len(intersection) / len(job_cat_skills)
                skill_scores[category] = score
            else:
                skill_scores[category] = 0.0
        
        return skill_scores
    
    def calculate_semantic_similarity(self, resume_text: str, job_text: str) -> float:
        """Calculate semantic similarity using multilingual sentence transformers"""
        if not self.sentence_model:
            return self.calculate_tfidf_similarity(resume_text, job_text)
        
        try:
            # Encode texts
            resume_embedding = self.sentence_model.encode([resume_text])
            job_embedding = self.sentence_model.encode([job_text])
            
            # Calculate cosine similarity
            similarity = cosine_similarity(resume_embedding, job_embedding)[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Erreur lors du calcul de similarité sémantique: {e}")
            return self.calculate_tfidf_similarity(resume_text, job_text)
    
    def calculate_tfidf_similarity(self, resume_text: str, job_text: str) -> float:
        """Fallback TF-IDF similarity calculation for French"""
        try:
            # Custom French stop words
            french_stop_words = list(self.stop_words)
            
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words=french_stop_words,
                ngram_range=(1, 3),
                min_df=1,
                max_df=0.8
            )
            
            corpus = [resume_text, job_text]
            tfidf_matrix = vectorizer.fit_transform(corpus)
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Erreur lors du calcul de similarité TF-IDF: {e}")
            return 0.0
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using French spaCy model"""
        if not self.nlp:
            return {}
        
        try:
            doc = self.nlp(text)
            entities = defaultdict(list)
            
            for ent in doc.ents:
                entities[ent.label_].append(ent.text)
            
            return dict(entities)
        except Exception as e:
            print(f"Erreur lors de l'extraction d'entités: {e}")
            return {}
    
    def calculate_ats_score(self, resume_path: str, job_path: str) -> Dict:
        """
        Main function to calculate comprehensive ATS matching score for French documents
        Returns detailed analysis and overall score
        """
        try:
            # Extract text from both sources
            if resume_path.lower().endswith('.pdf'):
                resume_text = self.extract_text_from_pdf(resume_path)
            else:
                resume_text = self.read_text_file(resume_path)
            
            job_text = self.read_text_file(job_path)
            
            if not resume_text or not job_text:
                return {
                    'erreur': 'Impossible d\'extraire le texte d\'une ou des deux sources',
                    'score_global': 0.0
                }
            
            # Preprocess texts
            resume_clean = self.preprocess_text(resume_text)
            job_clean = self.preprocess_text(job_text)
            
            # Extract features
            resume_skills = self.extract_skills(resume_text)
            job_skills = self.extract_skills(job_text)
            
            resume_education = self.extract_education(resume_text)
            job_education = self.extract_education(job_text)
            
            resume_experience = self.extract_experience_years(resume_text)
            job_experience = self.extract_experience_years(job_text)
            
            # Calculate various similarity scores
            semantic_similarity = self.calculate_semantic_similarity(resume_clean, job_clean)
            skill_matches = self.calculate_skill_match(resume_skills, job_skills)
            
            # Calculate overall skill score
            overall_skill_score = np.mean(list(skill_matches.values())) if skill_matches else 0.0
            
            # Calculate education match
            education_match = len(set(resume_education) & set(job_education)) / len(job_education) if job_education else 0.0
            
            # Calculate experience match
            experience_match = 1.0
            if job_experience and resume_experience:
                if resume_experience >= job_experience:
                    experience_match = 1.0
                else:
                    experience_match = resume_experience / job_experience
            elif job_experience and not resume_experience:
                experience_match = 0.0
            
            # Calculate weighted overall score
            weights = {
                'similarite_semantique': 0.3,
                'correspondance_competences': 0.4,
                'correspondance_formation': 0.15,
                'correspondance_experience': 0.15
            }
            
            overall_score = (
                weights['similarite_semantique'] * semantic_similarity +
                weights['correspondance_competences'] * overall_skill_score +
                weights['correspondance_formation'] * education_match +
                weights['correspondance_experience'] * experience_match
            )
            
            # Compile results
            results = {
                'score_global': round(overall_score, 4),
                'similarite_semantique': round(semantic_similarity, 4),
                'analyse_competences': {
                    'score_competences_global': round(overall_skill_score, 4),
                    'scores_par_categorie': {k: round(v, 4) for k, v in skill_matches.items()},
                    'competences_cv': resume_skills,
                    'competences_offre': job_skills
                },
                'correspondance_formation': round(education_match, 4),
                'analyse_experience': {
                    'score_correspondance': round(experience_match, 4),
                    'annees_experience_cv': resume_experience,
                    'annees_requises_offre': job_experience
                },
                'statistiques_texte': {
                    'nombre_mots_cv': len(resume_clean.split()),
                    'nombre_mots_offre': len(job_clean.split()),
                    'nombre_caracteres_cv': len(resume_text),
                    'nombre_caracteres_offre': len(job_text)
                },
                'recommandation': self.get_recommendation(overall_score)
            }
            
            return results
            
        except Exception as e:
            print(f"Erreur dans l'évaluation ATS: {e}")
            return {
                'erreur': str(e),
                'score_global': 0.0
            }
    
    def get_recommendation(self, score: float) -> str:
        """Get recommendation based on ATS score in French"""
        if score >= 0.8:
            return "Excellente correspondance - Fortement recommandé pour un entretien"
        elif score >= 0.6:
            return "Bonne correspondance - Recommandé pour examen"
        elif score >= 0.4:
            return "Correspondance correcte - À considérer avec d'autres facteurs"
        elif score >= 0.2:
            return "Correspondance faible - Peut ne pas convenir"
        else:
            return "Très faible correspondance - Non recommandé"

# Usage example
def evaluer_correspondance_cv_offre(chemin_cv: str, chemin_offre: str) -> Dict:
    """
    Évaluer la correspondance entre un CV et une offre d'emploi
    
    Args:
        chemin_cv: Chemin vers le fichier CV (PDF ou texte)
        chemin_offre: Chemin vers le fichier d'offre d'emploi (texte)
    
    Returns:
        Dictionnaire d'analyse complète
    """
    matcher = ATSMatcher()
    return matcher.calculate_ats_score(chemin_cv, chemin_offre)

# Example usage:
if __name__ == "__main__":
    # Exemple d'utilisation
    fichier_cv = "cv.pdf"
    fichier_offre = "offre_emploi.txt"
    
    resultats = evaluer_correspondance_cv_offre(fichier_cv, fichier_offre)
    
    print("=== RÉSULTATS DE CORRESPONDANCE ATS ===")
    print(f"Score Global: {resultats['score_global']:.2%}")
    print(f"Recommandation: {resultats['recommandation']}")
    print(f"Similarité Sémantique: {resultats['similarite_semantique']:.2%}")
    print(f"Score Compétences: {resultats['analyse_competences']['score_competences_global']:.2%}")
    print(f"Correspondance Formation: {resultats['correspondance_formation']:.2%}")
    print(f"Correspondance Expérience: {resultats['analyse_experience']['score_correspondance']:.2%}")