import os
from dados_treinamento.dados_de_treino import EXAMPLES
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import io
import random

# OpenAI
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# NLP & ML
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import download as nltk_download

# PDF reading
import PyPDF2

TEMPLATES = {
    "Produtivo": [
        "Olá, obrigado pelo contato. Recebemos sua solicitação e já iniciamos a análise.",
        "Sua mensagem foi recebida e nossa equipe já está cuidando do assunto. Em breve retornaremos com novidades.",
        "Estamos processando sua solicitação e logo mais enviaremos uma atualização."
    ],
    "Improdutivo": [
        "Olá! Agradecemos sua mensagem. Não é necessária nenhuma ação adicional no momento.",
        "Obrigado pelo contato. Estamos à disposição caso precise de algo.",
        "Sua mensagem foi recebida. Caso haja necessidade, entraremos em contato."
    ]
}

def classify_with_openai(text: str):
    #Tenta classificar o texto usando a API da OpenAI. Se falhar, a exceção deve ser capturada no chamador.
    prompt = f"""
    Classifique o seguinte email como 'Produtivo' ou 'Improdutivo'.
    Texto: {text}
    Responda apenas com uma dessas palavras.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # ou gpt-4.1, gpt-3.5-turbo etc.
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    category = response.choices[0].message.content.strip()
    if category not in ["Produtivo", "Improdutivo"]:
        category = "Improdutivo"  # default de segurança
    return category


def generate_response_local(category: str, email_text: str) -> str:
    # escolhe uma resposta aleatória dentro da categoria
    return random.choice(TEMPLATES.get(category, ["Obrigado pela sua mensagem."]))

# Ensure required NLTK data
nltk_download('stopwords')

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Simple preprocessing
STOPWORDS = set(stopwords.words('portuguese')) | set(stopwords.words('english'))
PS = PorterStemmer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\S+@\S+", ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [PS.stem(t) for t in text.split() if t not in STOPWORDS]
    return ' '.join(tokens)

texts = [clean_text(x[0]) for x in EXAMPLES]
labels = [1 if x[1] == 'Produtivo' else 0 for x in EXAMPLES]

VECT = TfidfVectorizer(ngram_range=(1,2), max_features=1000)
X = VECT.fit_transform(texts)
CLF = LogisticRegression()
CLF.fit(X, labels)

# Helpers
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def read_pdf(file_stream):
    try:
        reader = PyPDF2.PdfReader(file_stream)
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or '')
        return '\n'.join(text)
    except Exception:
        return ''

def classify_text(text: str) -> (str, float):
    cleaned = clean_text(text)
    x = VECT.transform([cleaned])
    prob = CLF.predict_proba(x)[0][1]
    label = 'Produtivo' if prob >= 0.5 else 'Improdutivo'
    return label, float(prob)

def fallback_response(category: str) -> str:
    if category == 'Produtivo':
        return ('Olá, obrigado pelo contato. Recebemos sua solicitação e já iniciamos a análise. '
                'Em breve retornaremos com uma atualização ou solicitaremos mais informações, se necessário.')
    else:
        return ('Olá! Agradecemos sua mensagem. Não é necessária nenhuma ação adicional no momento. '
                'Se precisar de algo específico, por favor nos informe.')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    text = request.form.get('email_text', '').strip()
    file = request.files.get('file')

    if not text and file and file.filename != '':
        filename = secure_filename(file.filename)
        if allowed_file(filename):
            ext = filename.rsplit('.', 1)[1].lower()
            if ext == 'txt':
                text = file.read().decode('utf-8', errors='ignore')
            elif ext == 'pdf':
                file.stream.seek(0)
                text = read_pdf(file)
            
            #SALVA OS ARQUIVOS EM UPLOADS
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        else:
            return redirect(url_for('index'))

    if not text:
        return render_template('index.html', error='Nenhum texto fornecido.')

    # Tenta com OpenAI, senão cai no local
    try:
        category = classify_with_openai(text)
        prob = None  # com OpenAI não vem probabilidade
    except Exception as e:
        print("Erro com OpenAI, caindo no modelo local:", e)
        category, prob = classify_text(text)

    suggested = generate_response_local(category, text)


    return render_template(
        'index.html',
        result=True,
        category=category,
        probability=prob,
        suggestion=suggested,
        original=text
    )


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
