# ChatPDF

ChatPDF is a PDF chat system based on RAG (Retrieval-Augmented Generation) technology. It allows users to upload PDF files and engage in interactive conversations with the document content.

## Features

- PDF file upload and processing
- Text extraction and chunking
- Vector database storage and retrieval
- LLaMA 2-based conversation generation
- Web interface for interaction

## File Structure
![13](https://github.com/user-attachments/assets/782df4da-a9a1-4351-aeab-61929997b86d)

- `openai_utils.py`: OpenAI API related utility functions
- `pdf_utils.py`: PDF processing tools
- `pdfchain.py`: Main logic for PDF processing
- `prompt_utils.py`: Prompt handling utilities
- `text_utils.py`: Text processing tools
- `vectordb_utils.py`: Vector database utilities
- `web_demo.py`: Web demonstration interface

## Requirements

Please refer to the `requirements.txt` file for the necessary Python dependencies.

## Usage

1. Clone the repository:
   ```
   git clone https://github.com/muzinan123/chatpdf.git
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the web demo:
   ```
   python web_demo.py
   ```

4. Open the displayed URL in your browser, upload a PDF file, and start the conversation.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Feel free to submit Pull Requests or create Issues.

## Notes

- Ensure you have sufficient computational resources to run the LLaMA 2 model.
- Please adhere to the terms of use and limitations of the AI models and APIs used.

## Acknowledgements

- [LLaMA 2](https://github.com/facebookresearch/llama)
- [OpenAI](https://openai.com/)
- All contributors to the open-source libraries used
