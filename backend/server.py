from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import traceback
import io
import PyPDF2
import tempfile
import uuid
from werkzeug.utils import secure_filename
from process import run_semantic_evaluation  # Import your existing functions
from ATSMatcher import ATSMatcher  # Import your ATS matcher class
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for browser extension

# Configure upload settings
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
UPLOAD_FOLDER = './uploads'
OUTPUT_FOLDER = './outputs'

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "message": "PDF processing server is running"})

@app.route('/process-pdf', methods=['POST'])
def process_pdf():
    """Main endpoint to process PDF with text overlay"""
    try:
        # Check if files are present
        if 'pdf_file' not in request.files or 'text_file' not in request.files:
            return jsonify({'error': 'Both PDF and text files are required'}), 400
        
        pdf_file = request.files['pdf_file']
        text_file = request.files['text_file']
        
        # Validate files
        if pdf_file.filename == '' or text_file.filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        if not pdf_file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'PDF file must have .pdf extension'}), 400
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded files
        pdf_filename = secure_filename(f"{session_id}_input.pdf")
        text_filename = secure_filename(f"{session_id}_text.txt")
        output_filename = f"{session_id}_output.pdf"
        
        pdf_path = os.path.join(UPLOAD_FOLDER, pdf_filename)
        text_path = os.path.join(UPLOAD_FOLDER, text_filename)
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        pdf_file.save(pdf_path)
        text_file.save(text_path)
        
        # Process the PDF using your existing function
        try:
            run_semantic_evaluation(pdf_path, text_path)
            
            # Check if output file was created
            if not os.path.exists(output_path):
                return jsonify({'error': 'Failed to create output PDF'}), 500
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'download_url': f'/download/{session_id}',
                'message': 'PDF processed successfully'
            })
            
        except Exception as e:
            return jsonify({'error': f'Processing failed: {str(e)}'}), 500
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/download/<session_id>', methods=['GET'])
def download_file(session_id):
    """Download the processed PDF"""
    try:
        output_filename = f"{session_id}_output.pdf"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        if not os.path.exists(output_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(
            output_path,
            as_attachment=True,
            download_name=f"processed_{session_id}.pdf",
            mimetype='application/pdf'
        )
        
    except Exception as e:
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@app.route('/cleanup/<session_id>', methods=['DELETE'])
def cleanup_files(session_id):
    """Clean up temporary files"""
    try:
        files_to_remove = [
            os.path.join(UPLOAD_FOLDER, f"{session_id}_input.pdf"),
            os.path.join(UPLOAD_FOLDER, f"{session_id}_text.txt"),
            os.path.join(OUTPUT_FOLDER, f"{session_id}_output.pdf")
        ]
        
        removed_count = 0
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                os.remove(file_path)
                removed_count += 1
        
        return jsonify({
            'success': True,
            'message': f'Cleaned up {removed_count} files'
        })
        
    except Exception as e:
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500






@app.route('/evaluate-semantic', methods=['POST'])
def evaluate_semantic():
    """Evaluate semantic similarity between CV and job description"""
    try:
        # Check if files are present
        if 'cv_file' not in request.files or 'job_description' not in request.files:
            return jsonify({'error': 'Both CV file and job description are required'}), 400
        
        cv_file = request.files['cv_file']
        job_desc_file = request.files['job_description']
        
        # Validate files
        if cv_file.filename == '' or job_desc_file.filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        if not cv_file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'CV file must have .pdf extension'}), 400
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Save uploaded files
        cv_filename = secure_filename(f"{session_id}_cv.pdf")
        job_desc_filename = secure_filename(f"{session_id}_job_desc.txt")
        
        cv_path = os.path.join(UPLOAD_FOLDER, cv_filename)
        job_desc_path = os.path.join(UPLOAD_FOLDER, job_desc_filename)
        
        cv_file.save(cv_path)
        job_desc_file.save(job_desc_path)
        
        # Run semantic evaluation
        try:
            # Import your evaluation functions
            from process import evaluate_semantic_distance, evaluate_semantic_distance_advanced
            
            # Run basic evaluation
            basic_results = evaluate_semantic_distance(cv_path, job_desc_path)
            
            # Run advanced evaluation if available
            try:
                advanced_results = evaluate_semantic_distance_advanced(cv_path, job_desc_path)
            except Exception as e:
                print(f"Advanced evaluation failed: {e}")
                advanced_results = None
            
            # Prepare response
            response_data = {
                'success': True,
                'session_id': session_id,
                'evaluation_results': {
                    'basic': basic_results if isinstance(basic_results, dict) else {'cosine_similarity': basic_results},
                    'advanced': advanced_results
                }
            }
            
            # Add interpretation
            if isinstance(basic_results, dict):
                similarity = basic_results.get('cosine_similarity', 0)
            else:
                similarity = basic_results if basic_results else 0
            
            # Generate interpretation
            if similarity > 0.8:
                interpretation = {
                    'level': 'very_high',
                    'message': 'Very high semantic similarity - CV is very well aligned with job description',
                    'recommendation': 'Excellent match! Your CV aligns very well with the job requirements.'
                }
            elif similarity > 0.6:
                interpretation = {
                    'level': 'high',
                    'message': 'High semantic similarity - CV is quite aligned with job description',
                    'recommendation': 'Good match! Consider minor adjustments to better highlight relevant skills.'
                }
            elif similarity > 0.4:
                interpretation = {
                    'level': 'moderate',
                    'message': 'Moderate semantic similarity - CV has some alignment with job description',
                    'recommendation': 'Fair match. Consider adding more relevant keywords and experiences from the job description.'
                }
            elif similarity > 0.2:
                interpretation = {
                    'level': 'low',
                    'message': 'Low semantic similarity - CV needs significant alignment with job description',
                    'recommendation': 'Poor match. Consider restructuring your CV to better match the job requirements.'
                }
            else:
                interpretation = {
                    'level': 'very_low',
                    'message': 'Very low semantic similarity - CV and job description are quite different',
                    'recommendation': 'Very poor match. This position may not be suitable, or your CV needs major revisions.'
                }
            
            response_data['interpretation'] = interpretation
            response_data['similarity_score'] = similarity
            
            # Clean up uploaded files after processing
            try:
                os.remove(cv_path)
                os.remove(job_desc_path)
            except:
                pass  # Don't fail if cleanup fails
            
            return jsonify(response_data)
            
        except Exception as e:
            return jsonify({'error': f'Evaluation failed: {str(e)}'}), 500
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500



# Required imports for the route

@app.route('/evaluate-ats', methods=['POST'])
def evaluate_ats():
    """Enhanced ATS evaluation endpoint using French ATS matching algorithm"""
    try:
        # Check if files are present
        if 'resume_file' not in request.files or 'job_description' not in request.files:
            return jsonify({'error': 'Both resume file and job description are required'}), 400
        
        resume_file = request.files['resume_file']
        job_desc_file = request.files['job_description']
        
        # Validate files
        if resume_file.filename == '' or job_desc_file.filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        if not resume_file.filename.lower().endswith('.pdf'):
            return jsonify({'error': 'Resume file must have .pdf extension'}), 400
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Extract text from PDF resume
        resume_text = ""
        try:
            # Read PDF file directly from memory
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(resume_file.read()))
            for page in pdf_reader.pages:
                resume_text += page.extract_text() + "\n"
        except Exception as e:
            return jsonify({'error': f'Failed to extract text from PDF: {str(e)}'}), 400
        # Read job description text
        try:
            job_desc_text = job_desc_file.read().decode('utf-8')
        except Exception as e:
            return jsonify({'error': f'Failed to read job description: {str(e)}'}), 400
        
        # Validate extracted content
        if not resume_text.strip() or not job_desc_text.strip():
            return jsonify({'error': 'Could not extract text from uploaded files'}), 400
        
        # Run ATS evaluation using French ATS matcher
        try:
            # Initialize French ATS matcher with error handling
            try:
                matcher = ATSMatcher()
            except Exception as init_error:
                print(f"Failed to initialize French ATS matcher: {init_error}")
                # Fallback to basic similarity calculation
                return perform_basic_similarity_evaluation(resume_text, job_desc_text, session_id)
            
            # Perform evaluation using text-based approach
            ats_results = perform_french_text_based_ats_evaluation(matcher, resume_text, job_desc_text)
            
            # Check if evaluation was successful
            if 'error' in ats_results or 'erreur' in ats_results:
                error_msg = ats_results.get('error', ats_results.get('erreur', 'Unknown error'))
                return jsonify({'error': f'ATS evaluation failed: {error_msg}'}), 500
            
            # Prepare response
            response_data = {
                'success': True,
                'session_id': session_id,
                'evaluation_results': {
                    'basic': ats_results if isinstance(ats_results, dict) else {'cosine_similarity': ats_results},
                    'advanced': ats_results
                }
            }
            
            # Add interpretation
            if isinstance(ats_results, dict):
                similarity = ats_results.get('score_global', ats_results.get('overall_score', 0))
            else:
                similarity = ats_results if ats_results else 0
            
            # Generate interpretation
            if similarity > 0.8:
                interpretation = {
                    'level': 'very_high',
                    'message': 'Very high semantic similarity - CV is very well aligned with job description',
                    'recommendation': 'Excellent match! Your CV aligns very well with the job requirements.'
                }
            elif similarity > 0.6:
                interpretation = {
                    'level': 'high',
                    'message': 'High semantic similarity - CV is quite aligned with job description',
                    'recommendation': 'Good match! Consider minor adjustments to better highlight relevant skills.'
                }
            elif similarity > 0.4:
                interpretation = {
                    'level': 'moderate',
                    'message': 'Moderate semantic similarity - CV has some alignment with job description',
                    'recommendation': 'Fair match. Consider adding more relevant keywords and experiences from the job description.'
                }
            elif similarity > 0.2:
                interpretation = {
                    'level': 'low',
                    'message': 'Low semantic similarity - CV needs significant alignment with job description',
                    'recommendation': 'Poor match. Consider restructuring your CV to better match the job requirements.'
                }
            else:
                interpretation = {
                    'level': 'very_low',
                    'message': 'Very low semantic similarity - CV and job description are quite different',
                    'recommendation': 'Very poor match. This position may not be suitable, or your CV needs major revisions.'
                }
            
            response_data['interpretation'] = interpretation
            response_data['similarity_score'] = similarity
            
            return jsonify(response_data)
            
        except Exception as e:
            print(f"ATS evaluation error: {e}")
            print(traceback.format_exc())
            return jsonify({'error': f'ATS evaluation failed: {str(e)}'}), 500
        
    except Exception as e:
        print(f"Server error in evaluate_ats: {e}")
        print(traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'}), 500


def perform_french_text_based_ats_evaluation(matcher, resume_text, job_desc_text):
    """
    Perform ATS evaluation using French text analysis with comprehensive error handling
    
    Args:
        matcher: ATSMatcherFrench instance
        resume_text: Extracted resume text
        job_desc_text: Job description text
    
    Returns:
        Dictionary with evaluation results
    """
    try:
        # Validate inputs
        if not resume_text or not job_desc_text:
            return {
                'error': 'Empty text provided',
                'score_global': 0.0,
                'overall_score': 0.0
            }
        
        # Preprocess texts with error handling
        try:
            resume_clean = matcher.preprocess_text(resume_text)
            job_clean = matcher.preprocess_text(job_desc_text)
        except Exception as e:
            print(f"Error in text preprocessing: {e}")
            # Use basic preprocessing as fallback
            resume_clean = resume_text.lower().strip()
            job_clean = job_desc_text.lower().strip()
        
        # Extract features with error handling
        try:
            resume_skills = matcher.extract_skills(resume_text)
            job_skills = matcher.extract_skills(job_desc_text)
        except Exception as e:
            print(f"Error extracting skills: {e}")
            resume_skills = {}
            job_skills = {}
        
        try:
            resume_education = matcher.extract_education(resume_text)
            job_education = matcher.extract_education(job_desc_text)
        except Exception as e:
            print(f"Error extracting education: {e}")
            resume_education = []
            job_education = []
        
        try:
            resume_experience = matcher.extract_experience_years(resume_text)
            job_experience = matcher.extract_experience_years(job_desc_text)
        except Exception as e:
            print(f"Error extracting experience: {e}")
            resume_experience = None
            job_experience = None
        
        # Calculate semantic similarity with fallback
        try:
            semantic_similarity = matcher.calculate_semantic_similarity(resume_clean, job_clean)
        except Exception as e:
            print(f"Error calculating semantic similarity: {e}")
            try:
                semantic_similarity = matcher.calculate_tfidf_similarity(resume_clean, job_clean)
            except Exception as e2:
                print(f"Error calculating TF-IDF similarity: {e2}")
                semantic_similarity = 0.0
        
        # Calculate skill matches with error handling
        try:
            skill_matches = matcher.calculate_skill_match(resume_skills, job_skills)
            overall_skill_score = np.mean(list(skill_matches.values())) if skill_matches else 0.0
        except Exception as e:
            print(f"Error calculating skill matches: {e}")
            skill_matches = {}
            overall_skill_score = 0.0
        
        # Calculate education match with error handling
        try:
            education_match = len(set(resume_education) & set(job_education)) / len(job_education) if job_education else 0.0
        except Exception as e:
            print(f"Error calculating education match: {e}")
            education_match = 0.0
        
        # Calculate experience match with error handling
        try:
            experience_match = 1.0
            if job_experience and resume_experience:
                if resume_experience >= job_experience:
                    experience_match = 1.0
                else:
                    experience_match = resume_experience / job_experience
            elif job_experience and not resume_experience:
                experience_match = 0.0
        except Exception as e:
            print(f"Error calculating experience match: {e}")
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
        
        # Get recommendation with error handling
        try:
            recommendation = matcher.get_recommendation(overall_score)
        except Exception as e:
            print(f"Error getting recommendation: {e}")
            recommendation = "Unable to generate recommendation"
        
        # Compile results - maintain both French and English keys for compatibility
        results = {
            'score_global': round(overall_score, 4),
            'overall_score': round(overall_score, 4),  # English key for compatibility
            'similarite_semantique': round(semantic_similarity, 4),
            'semantic_similarity': round(semantic_similarity, 4),  # English key for compatibility
            'analyse_competences': {
                'score_competences_global': round(overall_skill_score, 4),
                'scores_par_categorie': {k: round(v, 4) for k, v in skill_matches.items()},
                'competences_cv': resume_skills,
                'competences_offre': job_skills
            },
            'skill_analysis': {  # English key for compatibility
                'overall_skill_score': round(overall_skill_score, 4),
                'category_scores': {k: round(v, 4) for k, v in skill_matches.items()},
                'resume_skills': resume_skills,
                'job_skills': job_skills
            },
            'correspondance_formation': round(education_match, 4),
            'education_match': round(education_match, 4),  # English key for compatibility
            'analyse_experience': {
                'score_correspondance': round(experience_match, 4),
                'annees_experience_cv': resume_experience,
                'annees_requises_offre': job_experience
            },
            'experience_analysis': {  # English key for compatibility
                'match_score': round(experience_match, 4),
                'resume_years': resume_experience,
                'job_required_years': job_experience
            },
            'statistiques_texte': {
                'nombre_mots_cv': len(resume_clean.split()) if resume_clean else 0,
                'nombre_mots_offre': len(job_clean.split()) if job_clean else 0,
                'nombre_caracteres_cv': len(resume_text) if resume_text else 0,
                'nombre_caracteres_offre': len(job_desc_text) if job_desc_text else 0
            },
            'text_statistics': {  # English key for compatibility
                'resume_word_count': len(resume_clean.split()) if resume_clean else 0,
                'job_word_count': len(job_clean.split()) if job_clean else 0,
                'resume_char_count': len(resume_text) if resume_text else 0,
                'job_char_count': len(job_desc_text) if job_desc_text else 0
            },
            'recommandation': recommendation,
            'recommendation': recommendation,  # English key for compatibility
            'cosine_similarity': round(semantic_similarity, 4)  # For basic results compatibility
        }
        
        return results
        
    except Exception as e:
        print(f"Error in French text-based ATS evaluation: {e}")
        print(traceback.format_exc())
        return {
            'error': str(e),
            'score_global': 0.0,
            'overall_score': 0.0,
            'cosine_similarity': 0.0
        }


def perform_basic_similarity_evaluation(resume_text, job_desc_text, session_id):
    """
    Fallback function for basic similarity calculation when French ATS fails
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Basic preprocessing
        resume_clean = resume_text.lower().strip()
        job_clean = job_desc_text.lower().strip()
        
        # Calculate TF-IDF similarity
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        corpus = [resume_clean, job_clean]
        tfidf_matrix = vectorizer.fit_transform(corpus)
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Create basic response structure
        basic_results = {
            'overall_score': round(similarity, 4),
            'score_global': round(similarity, 4),
            'semantic_similarity': round(similarity, 4),
            'cosine_similarity': round(similarity, 4),
            'recommendation': 'Basic similarity analysis performed due to processing limitations'
        }
        
        response_data = {
            'success': True,
            'session_id': session_id,
            'evaluation_results': {
                'basic': basic_results,
                'advanced': basic_results
            }
        }
        
        # Add interpretation
        if similarity > 0.8:
            interpretation = {
                'level': 'very_high',
                'message': 'Very high semantic similarity - CV is very well aligned with job description',
                'recommendation': 'Excellent match! Your CV aligns very well with the job requirements.'
            }
        elif similarity > 0.6:
            interpretation = {
                'level': 'high',
                'message': 'High semantic similarity - CV is quite aligned with job description',
                'recommendation': 'Good match! Consider minor adjustments to better highlight relevant skills.'
            }
        elif similarity > 0.4:
            interpretation = {
                'level': 'moderate',
                'message': 'Moderate semantic similarity - CV has some alignment with job description',
                'recommendation': 'Fair match. Consider adding more relevant keywords and experiences from the job description.'
            }
        elif similarity > 0.2:
            interpretation = {
                'level': 'low',
                'message': 'Low semantic similarity - CV needs significant alignment with job description',
                'recommendation': 'Poor match. Consider restructuring your CV to better match the job requirements.'
            }
        else:
            interpretation = {
                'level': 'very_low',
                'message': 'Very low semantic similarity - CV and job description are quite different',
                'recommendation': 'Very poor match. This position may not be suitable, or your CV needs major revisions.'
            }
        
        response_data['interpretation'] = interpretation
        response_data['similarity_score'] = similarity
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in basic similarity evaluation: {e}")
        return jsonify({
            'error': 'Failed to perform evaluation',
            'success': False,
            'session_id': session_id
        }), 500

def perform_french_text_based_ats_evaluation(matcher, resume_text, job_desc_text):
    """
    Perform ATS evaluation using French text analysis
    
    Args:
        matcher: ATSMatcherFrench instance
        resume_text: Extracted resume text
        job_desc_text: Job description text
    
    Returns:
        Dictionary with evaluation results
    """
    try:
        # Preprocess texts
        resume_clean = matcher.preprocess_text(resume_text)
        job_clean = matcher.preprocess_text(job_desc_text)
        
        # Extract features
        resume_skills = matcher.extract_skills(resume_text)
        job_skills = matcher.extract_skills(job_desc_text)
        
        resume_education = matcher.extract_education(resume_text)
        job_education = matcher.extract_education(job_desc_text)
        
        resume_experience = matcher.extract_experience_years(resume_text)
        job_experience = matcher.extract_experience_years(job_desc_text)
        
        # Calculate various similarity scores
        semantic_similarity = matcher.calculate_semantic_similarity(resume_clean, job_clean)
        skill_matches = matcher.calculate_skill_match(resume_skills, job_skills)
        
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
        
        # Get recommendation
        recommendation = matcher.get_recommendation(overall_score)
        
        # Compile results - maintain both French and English keys for compatibility
        results = {
            'score_global': round(overall_score, 4),
            'overall_score': round(overall_score, 4),  # English key for compatibility
            'similarite_semantique': round(semantic_similarity, 4),
            'semantic_similarity': round(semantic_similarity, 4),  # English key for compatibility
            'analyse_competences': {
                'score_competences_global': round(overall_skill_score, 4),
                'scores_par_categorie': {k: round(v, 4) for k, v in skill_matches.items()},
                'competences_cv': resume_skills,
                'competences_offre': job_skills
            },
            'skill_analysis': {  # English key for compatibility
                'overall_skill_score': round(overall_skill_score, 4),
                'category_scores': {k: round(v, 4) for k, v in skill_matches.items()},
                'resume_skills': resume_skills,
                'job_skills': job_skills
            },
            'correspondance_formation': round(education_match, 4),
            'education_match': round(education_match, 4),  # English key for compatibility
            'analyse_experience': {
                'score_correspondance': round(experience_match, 4),
                'annees_experience_cv': resume_experience,
                'annees_requises_offre': job_experience
            },
            'experience_analysis': {  # English key for compatibility
                'match_score': round(experience_match, 4),
                'resume_years': resume_experience,
                'job_required_years': job_experience
            },
            'statistiques_texte': {
                'nombre_mots_cv': len(resume_clean.split()),
                'nombre_mots_offre': len(job_clean.split()),
                'nombre_caracteres_cv': len(resume_text),
                'nombre_caracteres_offre': len(job_desc_text)
            },
            'text_statistics': {  # English key for compatibility
                'resume_word_count': len(resume_clean.split()),
                'job_word_count': len(job_clean.split()),
                'resume_char_count': len(resume_text),
                'job_char_count': len(job_desc_text)
            },
            'recommandation': recommendation,
            'recommendation': recommendation,  # English key for compatibility
            'cosine_similarity': round(semantic_similarity, 4)  # For basic results compatibility
        }
        
        return results
        
    except Exception as e:
        print(f"Error in French text-based ATS evaluation: {e}")
        return {
            'error': str(e),
            'score_global': 0.0,
            'overall_score': 0.0
        }
    



if __name__ == '__main__':
    print("Starting PDF processing server...")
    print("Server will be available at: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
app.run(debug=True, host='0.0.0.0', port=5000)
