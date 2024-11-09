# Multi-LLM RAG Powered QA Bot

This project implements a Retrieval-Augmented Generation (RAG) chatbot using the Langchain framework and Azure OpenAI services. The chatbot can answer various questions based on context retrieved from a PDF document and allows users to explore previous interactions.

## Features

- **Multi-LLM Responses**: Utilizes two instances of Azure ChatGPT for parallel responses.
- **Retrieval-Augmented Generation**: Combines question-answering capabilities with document retrieval to provide accurate answers.
- **Gradio Interface**: User-friendly interface with two tabs for chat and previous responses.
- **Search Functionality**: Easily search through previously asked questions and their responses.

## Installation

Ensure you have Python 3.7 or higher installed. Then, install the required libraries using pip:

```bash
pip install --quiet --upgrade langchain langchain-community gradio pypdf langchain-openai faiss-cpu
```

## Configuration

1. Create an `azure_credentials.env` file to store your Azure OpenAI credentials:
   ```plaintext
   EMBEDDING_MODEL_NAME=your_embedding_model_name
   EMBEDDING_ENDPOINT=your_embedding_endpoint
   EMBEDDING_API_VERSION=your_embedding_api_version
   EMBEDDING_API_KEY=your_embedding_api_key
   AZURE_OPENAI_API_KEY=your_azure_openai_api_key
   API_VERSION=your_api_version
   AZURE_ENDPOINT=your_azure_endpoint
   ```

2. Replace the placeholders with your actual Azure OpenAI credentials.

## Usage

1. Prepare your PDF document by placing it in the project directory and updating the `file_path` variable in the script.
2. Run the `RAG.ipynb` notebook in Google Colab or locally to start the Gradio interface.
3. Use the **Chatbot** tab to ask questions and receive answers.
4. Use the **Previous Responses** tab to search through your interaction history.

## Project Structure

- **RAG.ipynb**: Main notebook containing the implementation of the chatbot.
- **.env**: Configuration file for sensitive information (API keys and endpoints).
- **book.pdf**: Sample PDF document used for context retrieval.

## Acknowledgments

This project leverages the following libraries:
- [Langchain](https://github.com/hwchase17/langchain): For managing language models and data retrieval.
- [Gradio](https://gradio.app/): For building user interfaces.
- [PyPDF](https://github.com/py-pdf/PyPDF2): For loading PDF documents.
