import os
from dados_treinamento.dados_de_treino import EXAMPLES
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import random
import time

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


def classify_with_openai(text: str):
    """
    Retorna a categoria e a resposta completa da OpenAI.
    """
    prompt = f"""
    Classifique o seguinte email como 'Produtivo' ou 'Improdutivo'.
    Considere que o contexto do email é uma grande empresa do setor financeiro.
    Texto: {text}
    Responda primeiro com a categoria, depois explique resumidamente sua decisão em no máximo 2 linhas.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    content = response.choices[0].message.content.strip()

    # Pega a categoria da primeira linha (ou default)
    first_line = content.splitlines()[0].strip()
    category = first_line if first_line in ["Produtivo", "Improdutivo"] else "Improdutivo"

    return category, content


# Baixando as stopwords no ambiente
nltk_download('stopwords')

# Define onde arquivos serão salvos e quais as extensões
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# Inicializa o Flask e cria a pasta uploads
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define lista de stopwords inglês e português e cria o stemmer
STOPWORDS = set(stopwords.words('portuguese')) | set(stopwords.words('english'))
PS = PorterStemmer()

# Remove emails, links, números e pontuações. Tudo em minúsculo. Aplica stemming
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\S+@\S+", ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [PS.stem(t) for t in text.split() if t not in STOPWORDS]
    return ' '.join(tokens)

# Trata os exemplos e atribui 1 para produtivo e 0 para improdutivo
texts = [clean_text(x[0]) for x in EXAMPLES]
labels = [1 if x[1] == 'Produtivo' else 0 for x in EXAMPLES]

# Cria pesos para as palavras para treinar o modelo usando os exemplos
VECT = TfidfVectorizer(ngram_range=(1,2), max_features=1000)
X = VECT.fit_transform(texts)
CLF = LogisticRegression()
CLF.fit(X, labels)

# Verifica se o arquivo tem extensão válida
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Extrai texto de todas as páginas de um PDF
def read_pdf(file_stream):
    try:
        reader = PyPDF2.PdfReader(file_stream)
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or '')
        return '\n'.join(text)
    except Exception:
        return ''

# Classifica usando o modelo local se a OPENAI não responder
def classify_text(text: str) -> (str, float):
    cleaned = clean_text(text)
    x = VECT.transform([cleaned])
    prob = CLF.predict_proba(x)[0][1]
    label = 'Produtivo' if prob >= 0.5 else 'Improdutivo'
    return label, float(prob)

# Retorna mensagens para cada categoria se a OPENAI não responder
def fallback_response(category: str) -> str:
    if category == 'Produtivo':
        return ('Olá, obrigado pelo contato. Recebemos sua solicitação e já iniciamos a análise. '
                'Em breve retornaremos com uma atualização ou solicitaremos mais informações, se necessário.')
    else:
        return ('Olá! Agradecemos sua mensagem. Não é necessária nenhuma ação adicional no momento. '
                'Se precisar de algo específico, por favor nos informe.')

# Rota inicial inicializando o index.html
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


def save_training_example(text: str, category: str, filename: str = None):
    """
    Salva o texto classificado dentro de uploads/Produtivo ou uploads/Improdutivo.
    Sempre gera nome único com timestamp, tanto para arquivos enviados
    quanto para textos colados.
    """
    folder = os.path.join(app.config['UPLOAD_FOLDER'], category)
    os.makedirs(folder, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if filename:  
        # mantém extensão se existir
        name, ext = os.path.splitext(filename)
        filename = f"{name}_{timestamp}{ext if ext else '.txt'}"
    else:
        filename = f"input_{timestamp}.txt"

    filepath = os.path.join(folder, filename)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(text)


@app.route('/process', methods=['POST'])
def process():
    #recebe texto ou arquivo digitado
    text = request.form.get('email_text', '').strip()
    file = request.files.get('file')
    filename = None

    # Se for upload de arquivo
    if not text and file and file.filename != '':
        filename = secure_filename(file.filename)
        if allowed_file(filename):
            ext = filename.rsplit('.', 1)[1].lower()
            if ext == 'txt':
                text = file.read().decode('utf-8', errors='ignore')
            elif ext == 'pdf':
                file.stream.seek(0)
                text = read_pdf(file)
        else:
            return redirect(url_for('index'))

    if not text:
        return render_template('index.html', error='Nenhum texto fornecido.')

    # Classificação
    try:
        # Tenta com OpenAI
        category, suggested = classify_with_openai(text)
        prob = None  # não usamos probabilidade aqui
    except Exception as e:
        print("Erro com OpenAI, caindo no modelo local:", e)
        category, prob = classify_text(text)
        suggested = fallback_response(category)

    # Salva no diretório correto, sem duplicar
    save_training_example(text, category, filename)

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
