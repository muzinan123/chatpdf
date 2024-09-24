#!/usr/bin/env python
# coding=utf-8
import gradio as gr
from openai_utils import get_completion
from prompt_utils import build_prompt
from vectordb_utils import InMemoryVecDB
from pdf_utils import extract_text_from_pdf
from text_utils import split_text

vec_db = InMemoryVecDB()


def init_db(file):
    paragraphs = extract_text_from_pdf(file.name)
    documents = split_text(paragraphs, 500, 100)
    vec_db.add_documents(documents)


def chat(user_input, chatbot, context, search_field):
    search_results = vec_db.search(user_input, 2)
    search_field = "\n\n".join(search_results)
    prompt = build_prompt(info=search_results, query=user_input)
    response = get_completion(prompt, context)
    chatbot.append((user_input, response))
    context.append({'role': 'user', 'content': user_input})
    context.append({'role': 'assistant', 'content': response})
    return "", chatbot, context, search_field


def reset_state():
    return [], [], "", ""


def main():
    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">ChatPDF</h1>""")

        with gr.Row():
            with gr.Column():
                fileCtrl = gr.File(label="Upload File", file_types=['.pdf'])

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot()
            with gr.Column(scale=2):
                # gr.HTML("""<h4>Search Results</h4>""")
                search_field = gr.Textbox(show_label=False, placeholder="Search results...", lines=10)
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=2)
                with gr.Row():
                    submitBtn = gr.Button("Submit", variant="primary")
                    emptyBtn = gr.Button("Clear")

        context = gr.State([])

        submitBtn.click(chat, [user_input, chatbot, context, search_field],
                        [user_input, chatbot, context, search_field])
        emptyBtn.click(reset_state, outputs=[chatbot, context, user_input, search_field])

        fileCtrl.upload(init_db, inputs=[fileCtrl])

    demo.queue().launch(share=False, server_name='0.0.0.0', server_port=8888, inbrowser=True)


if __name__ == "__main__":
    main()
