from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import tempfile
import uuid
from werkzeug.utils import secure_filename
from process import run_semantic_evaluation  # Import your existing functions

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

@app.route('/evaluate-semantic-text', methods=['POST'])
def evaluate_semantic_text():
    """Evaluate semantic similarity using text content directly (alternative endpoint)"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        cv_text = data.get('cv_text', '')
        job_desc_text = data.get('job_description_text', '')
        
        if not cv_text or not job_desc_text:
            return jsonify({'error': 'Both cv_text and job_description_text are required'}), 400
        
        # Generate unique session ID
        session_id = str(uuid.uuid4())
        
        # Create temporary files
        cv_filename = f"{session_id}_cv_temp.txt"
        job_desc_filename = f"{session_id}_job_desc_temp.txt"
        
        cv_path = os.path.join(UPLOAD_FOLDER, cv_filename)
        job_desc_path = os.path.join(UPLOAD_FOLDER, job_desc_filename)
        
        # Write text to temporary files
        with open(cv_path, 'w', encoding='utf-8') as f:
            f.write(cv_text)
        
        with open(job_desc_path, 'w', encoding='utf-8') as f:
            f.write(job_desc_text)
        
        # For text-based evaluation, we'll use TF-IDF similarity
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
            
            # Preprocess texts
            def preprocess_text(text):
                text = text.lower()
                text = ' '.join(text.split())
                return text
            
            cv_text_clean = preprocess_text(cv_text)
            job_desc_clean = preprocess_text(job_desc_text)
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            corpus = [cv_text_clean, job_desc_clean]
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            similarity_score = similarity_matrix[0][0]
            
            # Calculate word overlap
            def calculate_word_overlap(text1, text2):
                words1 = set(text1.split())
                words2 = set(text2.split())
                intersection = words1.intersection(words2)
                union = words1.union(words2)
                return len(intersection) / len(union) if union else 0
            
            word_overlap = calculate_word_overlap(cv_text_clean, job_desc_clean)
            
            # Length comparison
            cv_length = len(cv_text_clean.split())
            job_desc_length = len(job_desc_clean.split())
            length_ratio = min(cv_length, job_desc_length) / max(cv_length, job_desc_length) if max(cv_length, job_desc_length) > 0 else 0
            
            results = {
                'cosine_similarity': similarity_score,
                'semantic_distance': 1 - similarity_score,
                'word_overlap': word_overlap,
                'length_similarity': length_ratio,
                'cv_word_count': cv_length,
                'job_desc_word_count': job_desc_length
            }
            
            # Generate interpretation
            if similarity_score > 0.8:
                interpretation = {
                    'level': 'very_high',
                    'message': 'Very high semantic similarity - CV is very well aligned with job description',
                    'recommendation': 'Excellent match! Your CV aligns very well with the job requirements.'
                }
            elif similarity_score > 0.6:
                interpretation = {
                    'level': 'high',
                    'message': 'High semantic similarity - CV is quite aligned with job description',
                    'recommendation': 'Good match! Consider minor adjustments to better highlight relevant skills.'
                }
            elif similarity_score > 0.4:
                interpretation = {
                    'level': 'moderate',
                    'message': 'Moderate semantic similarity - CV has some alignment with job description',
                    'recommendation': 'Fair match. Consider adding more relevant keywords and experiences from the job description.'
                }
            elif similarity_score > 0.2:
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
            
            response_data = {
                'success': True,
                'session_id': session_id,
                'evaluation_results': results,
                'interpretation': interpretation,
                'similarity_score': similarity_score
            }
            
            # Clean up temporary files
            try:
                os.remove(cv_path)
                os.remove(job_desc_path)
            except:
                pass
            
            return jsonify(response_data)
            
        except Exception as e:
            return jsonify({'error': f'Text evaluation failed: {str(e)}'}), 500
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500






if __name__ == '__main__':
    print("Starting PDF processing server...")
    print("Server will be available at: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    app.run(debug=True, host='0.0.0.0', port=5000)