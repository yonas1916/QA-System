from flask import Flask, request, render_template_string, jsonify
import pypdf
import re
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import sent_tokenize
import os
from pathlib import Path

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

app = Flask(__name__)
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Global variable to store PDF data
pdf_chunks = []
bm25 = None

def extract_section_heading(page_num, lines):
    """Extract or infer section heading from page"""
    for line in lines[:10]:
        line = line.strip()
        if not line:
            continue
        if line.isupper() and len(line) < 60:
            return line
        if line.endswith(':') and len(line) < 60:
            return line
        if re.match(r'^(\d+\.\d+|\d+\.|Chapter|Section|Definition|What is)', line, re.I):
            return line
    if lines and len(lines[0].strip()) < 60:
        return lines[0].strip()[:50]
    return f"Page {page_num} content"

def load_pdf(pdf_path):
    global pdf_chunks, bm25
    pdf_chunks = []
    
    with open(pdf_path, 'rb') as file:
        reader = pypdf.PdfReader(file)
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text()
            if not text:
                continue
            lines = text.split('\n')
            section = extract_section_heading(page_num, lines)
            pdf_chunks.append({
                'page_num': page_num,
                'section': section,
                'text': text
            })
    
    # Build BM25 index
    if pdf_chunks:
        tokenized_chunks = [re.findall(r'\b\w+\b', chunk['text'].lower()) for chunk in pdf_chunks]
        bm25 = BM25Okapi(tokenized_chunks)
    return len(pdf_chunks)

def find_answer(question):
    global bm25, pdf_chunks
    
    if not bm25 or not pdf_chunks:
        return {
            'answer': "No PDF loaded. Please upload a PDF first.",
            'page': None,
            'section': None
        }
    
    tokenized_question = re.findall(r'\b\w+\b', question.lower())
    scores = bm25.get_scores(tokenized_question)
    
    # Find best matching chunk
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    
    if scores[best_idx] == 0:
        return {
            'answer': "Sorry, I couldn't find an answer to your question in the PDF.",
            'page': None,
            'section': None
        }
    
    chunk = pdf_chunks[best_idx]
    
    # Find best sentence in the chunk
    sentences = sent_tokenize(chunk['text'])
    if not sentences:
        return {
            'answer': chunk['text'][:500],
            'page': chunk['page_num'],
            'section': chunk['section']
        }
    
    question_tokens = set(tokenized_question)
    best_sentence = sentences[0]
    best_score = 0
    
    for sentence in sentences:
        sentence_tokens = set(re.findall(r'\b\w+\b', sentence.lower()))
        overlap = len(question_tokens.intersection(sentence_tokens))
        if overlap > best_score:
            best_score = overlap
            best_sentence = sentence
    
    return {
        'answer': best_sentence,
        'page': chunk['page_num'],
        'section': chunk['section']
    }

# HTML Template
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>PDF QA System</title>
    <meta charset="UTF-8">
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 { margin: 0 0 10px 0; }
        .content { padding: 30px; }
        .upload-area {
            border: 2px dashed #667eea;
            border-radius: 10px;
            padding: 40px;
            text-align: center;
            background: #f8f9ff;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 20px;
        }
        .upload-area:hover {
            background: #f0f2ff;
            border-color: #764ba2;
        }
        .file-info {
            background: #e8ebff;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
            display: none;
        }
        .file-info.show { display: block; }
        .question-input {
            width: 100%;
            padding: 15px;
            font-size: 16px;
            border: 2px solid #ddd;
            border-radius: 10px;
            font-family: inherit;
            resize: vertical;
        }
        .question-input:focus {
            outline: none;
            border-color: #667eea;
        }
        .ask-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            font-size: 16px;
            border-radius: 10px;
            cursor: pointer;
            margin-top: 10px;
        }
        .ask-btn:hover { transform: translateY(-2px); }
        .answer-area {
            background: #f8f9ff;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            display: none;
        }
        .answer-area.show { display: block; }
        .answer-text {
            background: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 15px;
            line-height: 1.6;
        }
        .metadata {
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        .metadata-item {
            background: white;
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 14px;
        }
        .loading {
            text-align: center;
            padding: 20px;
            display: none;
        }
        .loading.show { display: block; }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📄 PDF QA System</h1>
            <p>Upload a PDF, ask questions, get answers with page references</p>
        </div>
        <div class="content">
            <div class="upload-area" id="uploadArea">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#667eea" stroke-width="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
                <p style="margin-top: 10px;">Click or drag & drop PDF here</p>
                <input type="file" id="fileInput" accept=".pdf" style="display: none;">
            </div>
            
            <div class="file-info" id="fileInfo">
                <strong>📄 Current PDF:</strong> <span id="fileName"></span>
                <span style="color: green;">✓ Loaded</span>
            </div>
            
            <textarea id="questionInput" class="question-input" rows="3" placeholder="Type your question here... (e.g., What is information retrieval?)"></textarea>
            <button class="ask-btn" id="askBtn">🔍 Ask Question</button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Searching for answer...</p>
            </div>
            
            <div class="answer-area" id="answerArea">
                <h3>📖 Answer</h3>
                <div class="answer-text" id="answerText"></div>
                <div class="metadata">
                    <div class="metadata-item">📍 Page: <strong id="pageNum">-</strong></div>
                    <div class="metadata-item">📑 Section: <strong id="sectionName">-</strong></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const fileInfo = document.getElementById('fileInfo');
        const fileName = document.getElementById('fileName');
        const askBtn = document.getElementById('askBtn');
        const questionInput = document.getElementById('questionInput');
        const loading = document.getElementById('loading');
        const answerArea = document.getElementById('answerArea');
        const answerText = document.getElementById('answerText');
        const pageNum = document.getElementById('pageNum');
        const sectionName = document.getElementById('sectionName');
        
        uploadArea.onclick = () => fileInput.click();
        
        fileInput.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            
            const formData = new FormData();
            formData.append('file', file);
            
            loading.classList.add('show');
            
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            loading.classList.remove('show');
            
            if (response.ok) {
                const data = await response.json();
                fileName.textContent = file.name;
                fileInfo.classList.add('show');
                alert(data.message);
            } else {
                alert('Upload failed');
            }
        };
        
        askBtn.onclick = async () => {
            const question = questionInput.value.trim();
            if (!question) {
                alert('Please enter a question');
                return;
            }
            
            loading.classList.add('show');
            answerArea.classList.remove('show');
            
            const response = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ question: question })
            });
            
            loading.classList.remove('show');
            
            if (response.ok) {
                const data = await response.json();
                answerText.textContent = data.answer;
                pageNum.textContent = data.page || 'Not found';
                sectionName.textContent = data.section || 'Not found';
                answerArea.classList.add('show');
            } else {
                const error = await response.json();
                alert('Error: ' + (error.error || 'Unknown error'));
            }
        };
        
        questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                askBtn.click();
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/upload', methods=['POST'])
def upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.endswith('.pdf'):
            return jsonify({'error': 'Only PDF files allowed'}), 400
        
        file_path = UPLOAD_DIR / file.filename
        file.save(file_path)
        
        try:
            num_pages = load_pdf(str(file_path))
            os.remove(file_path)
            return jsonify({'message': f'PDF loaded! {num_pages} pages ready.'})
        except Exception as e:
            if file_path.exists():
                os.remove(file_path)
            return jsonify({'error': str(e)}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON'}), 400
        
        question = data.get('question', '').strip()
        if not question:
            return jsonify({'error': 'Question cannot be empty'}), 400
        
        result = find_answer(question)
        return jsonify(result)
    except Exception as e:
        print(f"Error in /ask: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print('=' * 50)
    print(' PDF QA System Started!')
    print(' Open browser and go to: http://localhost:5000')
    print('=' * 50)
    app.run(debug=True, host='0.0.0.0', port=5000)
