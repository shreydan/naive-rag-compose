import gradio as gr
import httpx
from openai import OpenAI
import os

client = OpenAI(
    base_url=os.environ['LLM_URL'],
    api_key=os.environ['LLM_KEY'],
)

async def upload_file(file_obj, new=True):
    url = os.environ['UPLOAD_ENDPOINT']
    file_path = file_obj.name

    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_obj.name, f)}
            data = {"new": str(new).lower()}
            async with httpx.AsyncClient(timeout=60) as client_http:
                response = await client_http.post(url, data=data, files=files)
                response.raise_for_status()
                try:
                    return f"Status code: {response.status_code}\nResponse: {response.json()}"
                except Exception:
                    return f"Status code: {response.status_code}\nRaw Response: {response.text}"
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except httpx.HTTPError as e:
        return f"HTTP Error: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

async def get_context(query):
    url = f"{os.environ['QUERY_ENDPOINT']}/{query}"

    rag_prompt = """You're an AI Assistant Chatbot. Please answer user queries
based on the provided context. Do not answer if it is out of the context.
Understand the context carefully and answer accordingly."""

    try:
        async with httpx.AsyncClient() as client_http:
            response = await client_http.get(url)
            response = response.json()
            context = response.get('context', '')
            return f"{rag_prompt}\n\n{context}\n\nUser Query:\n{query}"
    except Exception:
        return f"{rag_prompt}\n\nNo context found.\n\nUser Query:\n{query}"

async def respond(message, history, file_obj, new):
    if file_obj:
        upload_result = await upload_file(file_obj, new)
        new_history = history + [{"role": "assistant", "content": "File uploaded successfully. You may now chat with it."}]
        yield new_history
        return

    if message.strip():
        new_history = history + [{"role": "user", "content": message}]
        yield new_history

        rag_context = await get_context(message)
        messages = [{"role": "user", "content": rag_context}] + new_history

        try:
            completion = client.chat.completions.create(
                model="google/gemma-3n-e4b-it:free",
                messages=messages,
                stream=True
            )

            partial_response = ""
            for chunk in completion:
                delta = chunk.choices[0].delta
                if delta.content:
                    partial_response += delta.content
                    updated_history = new_history + [{"role": "assistant", "content": partial_response}]
                    yield updated_history

        except Exception as e:
            yield new_history + [{"role": "assistant", "content": f"Error: {str(e)}"}]

def clear_chat():
    return []

with gr.Blocks() as demo:
    gr.Markdown("## 🤖 RAG Chat Assistant", elem_classes="text-center")

    chatbot = gr.Chatbot(label=None, height=450, type="messages")

    with gr.Row(equal_height=True):
        textbox = gr.Textbox(
            placeholder="Type your message here...",
            scale=4,
            container=True,
            lines=1,
            autofocus=True
        )
        submit_btn = gr.Button("Send", scale=1)

    with gr.Row():
        file_input = gr.File(label="", file_types=[".txt", ".md"])
        new_checkbox = gr.Checkbox(label="New File", value=True)
        clear_btn = gr.Button("Clear Chat")

    # Text submit
    submit_btn.click(
        fn=respond,
        inputs=[textbox, chatbot, gr.State(None), gr.State(None)],
        outputs=[chatbot],
        show_progress=True
    ).then(lambda: "", outputs=[textbox])

    textbox.submit(
        fn=respond,
        inputs=[textbox, chatbot, gr.State(None), gr.State(None)],
        outputs=[chatbot],
        show_progress=True
    ).then(lambda: "", outputs=[textbox])

    # File upload
    file_input.upload(
        fn=respond,
        inputs=[gr.State(""), chatbot, file_input, new_checkbox],
        outputs=[chatbot],
        show_progress=True
    ).then(lambda: None, outputs=[file_input])

    clear_btn.click(fn=clear_chat, outputs=[chatbot])

if __name__ == "__main__":
    demo.launch()
