import gradio as gr
from ktem.app import BasePage


class ChatPanel(BasePage):
    def __init__(self, app):
        self._app = app
        self.on_building_ui()

    def on_building_ui(self):
        welcome_message = (
            "👋 **Welcome to DLGPT Assistant!**\n\n"
            "Need help with AOSP on Datalogic devices? You're in the right place!\n\n"
            "**How to Get Practical Code Assistance:**\n"
            "- **Be Straight & Specific:** Ask detailed questions. *Ex:* \"Provide Java code to implement NFC scanning on Skorpio X5.\"\n"
            "- **Mention the Device & Context:** Specify which device and any relevant details. *Ex:* \"I need Android code to handle barcode scanning on Memor 20.\"\n"
            "- **State Your Goal Clearly:** Explain what you're trying to achieve. *Ex:* \"How to build scanner app that use Datalogic SDK?\"\n\n"
            "**Important:**\n"
            "- Use **`Code Search`**, **`File Collection`**, or **`GraphRAG Collection`** separately in your queries. **Do not combine them in the same query**, as it may overload the system and lead to incomplete answers.\n\n"
            "For more assistance, visit the **Help** tab."
        )
        self.chatbot = gr.Chatbot(
            label=self._app.app_name,
            placeholder=welcome_message,
            show_label=False,
            elem_id="main-chat-bot",
            show_copy_button=True,
            likeable=True,
            bubble_full_width=False,
        )
        with gr.Row():
            self.text_input = gr.MultimodalTextbox(
                interactive=True,
                scale=20,
                file_count="multiple",
                placeholder="Chat input",
                container=False,
                show_label=False,
            )
            self.submit_btn = gr.Button(
                value="Send",
                scale=1,
                min_width=10,
                variant="primary",
                elem_classes=["cap-button-height"],
            )
            self.regen_btn = gr.Button(
                value="Regen",
                scale=1,
                min_width=10,
                elem_classes=["cap-button-height"],
            )
            self.stop_btn = gr.Button(  # Added Stop button
                value="Stop",
                scale=1,
                min_width=10,
                variant="secondary",
                elem_classes=["cap-button-height"],
            )

    def submit_msg(self, chat_input, chat_history):
        """Submit a message to the chatbot"""
        return "", chat_history + [(chat_input, None)]
