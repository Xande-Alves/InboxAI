# 📧 InboxAI - Sua caixa de entrada inteligente

Este projeto é uma API em Flask que classifica emails como Produtivo ou Improdutivo, em um contexto de uma empresa do setor financeiro, utilizando dois modos:

→ OpenAI (GPT-4o-mini): Classificação principal (se disponível).

→ Modelo local (Logistic Regression + TF-IDF): Usado como fallback quando a API da OpenAI não responde.

→ Também é possível enviar arquivos .txt ou .pdf para classificação.

---

## Funcionalidades

→ Classificação de emails em Produtivo ou Improdutivo.

→ Suporte a upload de arquivos TXT e PDF.

→ Treinamento local com exemplos armazenados em dados_treinamento/dados_de_treino.py.

→ Salva automaticamente os textos classificados em pastas organizadas (uploads/Produtivo e uploads/Improdutivo). Esses arquivos podem ser usados para um treinamento supervisionado do modelo para aumentar sua acurácia. 

→ Respostas personalizadas quando o modelo local é usado.

---

## Link da aplicação
O projeto pode ser rodado diretamente, sem necessidade de nenhuma instação, no link: https://inboxai-h0vw.onrender.com

---

## Pré-requisitos (para rodar localmente)

Antes de rodar o projeto, instale:

→ Python 3.9+

→ pip (gerenciador de pacotes Python)

Também será necessário uma API Key da OpenAI (caso queira usar GPT).

---

## Instalação

1. Clone este repositório e entre na pasta:
```bash
git clone https://github.com/Xande-Alves/InboxAI.git
cd InboxAI
```
2. Crie um ambiente virtual:
```bash
python -m venv venv
source venv/Scripts/activate
```
3. Instale as dependências:
```bash
pip install -r requirements.txt
```
4. Informe sua chave de API da OpenAI:
 ```bash
export OPENAI_API_KEY="SUA CHAVE AQUI"
```
5. Inicie o servidor:
```bash
python app.py
```

> A aplicação estará disponível em http://127.0.0.1:5000/

---

## Endpoints
### GET /

→ Retorna a página inicial (index.html) com o formulário para inserir texto ou enviar arquivos.

### POST /process

→ Processa o email informado no formulário ou no arquivo enviado.

→ Classifica como Produtivo ou Improdutivo.

→ Exibe a resposta sugerida (da OpenAI ou do modelo local).

Parâmetros:

→ email_text → Texto do email (campo do formulário).

→ file → Upload de arquivo .txt ou .pdf.

Retorno (HTML renderizado):

→ Categoria (Produtivo ou Improdutivo).

→ Probabilidade (se classificado pelo modelo local).

→ Sugestão de resposta.

Texto original processado.

---

## Estrutura do Projeto
```
.
├── app.py                        # Código principal da API
├── dados_treinamento/
│   └── dados_de_treino.py        # Exemplos de treino
├── static                        # Estilo e imagens
│   ├── logo2.jpg
│   └── style.css
├── templates/
│   └── index.html                # Frontend
├── uploads/                      # Emails salvos
│   ├── Produtivo/
│   └── Improdutivo/
├── venv                          # Ambiente virtual
├── requirements.txt              # Dependências do projeto
└── README.md                     # Este arquivo
```

---

## Tecnologias Utilizadas
→ Flask: Framework web

→ OpenAI Python: API GPT

→ scikit-learn: Machine Learning local

→ NLTK: Pré-processamento de texto

→ PyPDF2: Leitura de arquivos PDF

---

## Autor

Desenvolvido por Alexandre Alves