from flask import Flask, request, jsonify
from langchain_community.document_loaders import YoutubeLoader
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
import os
import logging
from dotenv import load_dotenv
from flask_cors import CORS

# Load environment variables
load_dotenv()
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app) 

# Initialize the language model
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

# Define the prompt template
prompt_template = """
Please provide a concise and informative summary of the content found at the following URL in {language}. The summary should be approximately 300 words and should highlight the main points, key arguments, and any significant conclusions or insights presented in the content. Ensure that the summary is clear and easy to understand for someone who has not accessed the original content.

URL Content:
{text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text", "language"])
summarize_chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)

# Add this for Vercel
@app.route('/')
def home():
    return "YouTube Summarizer API is running!"

@app.route('/summarize', methods=['POST'])
def summarize_youtube_video():
    try:
        data = request.get_json()
        youtube_url = data.get('youtube_url')
        language = data.get('language', 'English')  # Default to English if not specified
        
        if not youtube_url:
            return jsonify({"error": "YouTube URL is required"}), 400

        # Load YouTube video
        loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
        docs = loader.load()

        # Generate summary
        summary = summarize_chain.run({"input_documents": docs, "language": language})

        return jsonify({"summary": summary})

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": "An internal error occurred"}), 500

# Add this line at the end of the file:
app = app