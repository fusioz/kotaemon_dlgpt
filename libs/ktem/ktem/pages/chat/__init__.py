import asyncio
import re
import os
from copy import deepcopy
from typing import List, Optional

import gradio as gr
import requests
import urllib.parse
import markdown
from bs4 import BeautifulSoup
from ktem.app import BasePage
from ktem.components import reasonings
from ktem.db.models import Conversation, engine
from ktem.index.file.ui import File
from ktem.reasoning.prompt_optimization.suggest_conversation_name import SuggestConvNamePipeline
from ktem.reasoning.prompt_optimization.suggest_followup_chat import SuggestFollowupQuesPipeline
from plotly.io import from_json
from sqlmodel import Session, select
from theflow.settings import settings as flowsettings

from kotaemon.base import Document
from kotaemon.indices.ingests.files import KH_DEFAULT_FILE_EXTRACTORS

from ...utils import SUPPORTED_LANGUAGE_MAP
from .chat_panel import ChatPanel
from .common import STATE
from .control import ConversationControl
from .report import ReportIssue

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

# Constants
DEFAULT_SETTING = "(default)"
INFO_PANEL_SCALES = {True: 8, False: 4}

# JavaScript for PDF view modal
PDFVIEW_JS = """
function() {
    var links = document.getElementsByClassName("pdf-link");
    for (var i = 0; i < links.length; i++) {
        links[i].onclick = openModal;
    }

    var mindmap_el = document.getElementById('mindmap');
    if (mindmap_el) {
        var output = svgPanZoom(mindmap_el);
    }

    var link = document.getElementById("mindmap-toggle");
    if (link) {
        link.onclick = function(event) {
            event.preventDefault(); // Prevent the default link behavior
            var div = document.getElementById("mindmap-wrapper");
            if (div) {
                var currentHeight = div.style.height;
                if (currentHeight === '400px') {
                    var contentHeight = div.scrollHeight;
                    div.style.height = contentHeight + 'px';
                } else {
                    div.style.height = '400px'
                }
            }
        };
    }

    return [links.length]
}
"""

class ChatPage(BasePage):
    EXCLUDED_EXTENSIONS = {'.jar', '.o', '.class', '.exe', '.dll', '.so', '.bin', '.apk', '.aar'}

    def __init__(self, app):
        self._app = app
        self._indices_input = []
        self.projects = [
            "01-Memor20-android-11", "02-SkorpioX5-android-11", "03-Joya-android-11",
            "04-SkorpioX5-android-10", "Joya-android-13", "M20-A9", "Memor20-android-13",
            "Skorpio-X5-android-13", "android-14-r54", "android-datalogic-common",
            "luca_qcm4490", "opengrok", "stm32-jta11",
        ]

        self.nlp = spacy.load('en_core_web_sm')
        self.on_building_ui()

        # Initialize Gradio states
        self._preview_links = gr.State(value=None)
        self._reasoning_type = gr.State(value=None)
        self._llm_type = gr.State(value=None)
        self._conversation_renamed = gr.State(value=False)
        self._suggestion_updated = gr.State(value=False)
        self._info_panel_expanded = gr.State(value=True)

        # Initialize stop flags dictionary
        self.stop_flags = {}

    def on_building_ui(self):
        with gr.Row():
            # Initialize chat states
            self.state_chat = gr.State(STATE)
            self.state_retrieval_history = gr.State([])
            self.state_plot_history = gr.State([])
            self.state_plot_panel = gr.State(None)
            self.state_follow_up = gr.State(None)

            # Conversation Settings Panel
            with gr.Column(scale=1, elem_id="conv-settings-panel") as self.conv_column:
                self.chat_control = ConversationControl(self._app)
                self._build_code_search_options()
                self._build_index_collections()
                self._build_quick_upload()
                self.report_issue = ReportIssue(self._app)

            # Chat Area
            with gr.Column(scale=6, elem_id="chat-area"):
                self.chat_panel = ChatPanel(self._app)
                self._build_chat_settings()

            # Information Panel
            with gr.Column(scale=INFO_PANEL_SCALES[False], elem_id="chat-info-panel") as self.info_column:
                with gr.Accordion(label="ℹ️ Information Panel", open=True):
                    self.modal = gr.HTML("<div id='pdf-modal'></div>")
                    self.plot_panel = gr.Plot(visible=False)
                    self.info_panel = gr.HTML(elem_id="html-info-panel")

    def _build_code_search_options(self):
        """Builds the Code Search Options accordion."""
        with gr.Accordion(label="📁 Code Search Options", open=False):
            with gr.Column():
                self.enable_code_search = gr.Checkbox(label="Enable Code Search", value=True)
                self.search_projects = gr.CheckboxGroup(choices=self.projects, label="Select Projects to Search")
                self.search_defs = gr.Textbox(label="Definitions", placeholder="Enter definitions to include in the search...", lines=1)
                self.search_refs = gr.Textbox(label="References", placeholder="Enter references to include in the search...", lines=1)
                self.search_path = gr.Textbox(label="Path", placeholder="Enter path to include in the search...", lines=1)
                self.search_hist = gr.Textbox(label="History", placeholder="Enter history to include in the search...", lines=1)
                self.search_type = gr.Textbox(label="Type", placeholder="Enter type to include in the search...", lines=1)

    def _build_index_collections(self):
        """Builds the Index Collections accordions."""
        for index_id, index in enumerate(self._app.index_manager.indices):
            index.selector = None
            index_ui = index.get_selector_component_ui()
            if not index_ui:
                continue

            index_ui.unrender()
            with gr.Accordion(label=f"📄 {index.name} Collection", open=index_id < 1):
                index_ui.render()
                gr_index = index_ui.as_gradio_component()
                if gr_index:
                    if isinstance(gr_index, list):
                        index.selector = tuple(range(len(self._indices_input), len(self._indices_input) + len(gr_index)))
                        index.default_selector = index_ui.default()
                        self._indices_input.extend(gr_index)
                    else:
                        index.selector = len(self._indices_input)
                        index.default_selector = index_ui.default()
                        self._indices_input.append(gr_index)
                setattr(self, f"_index_{index.id}", index_ui)

    def _build_quick_upload(self):
        """Builds the Quick Upload accordion if indices exist."""
        if self._app.index_manager.indices:
            with gr.Accordion(label="📤 Quick Upload"):
                self.quick_file_upload = File(
                    file_types=list(KH_DEFAULT_FILE_EXTRACTORS.keys()),
                    file_count="multiple",
                    container=True,
                    show_label=False,
                )
                self.quick_file_upload_status = gr.Markdown()

    def _build_chat_settings(self):
        """Builds the Chat Settings accordion."""
        with gr.Row():
            with gr.Accordion(label="⚙️ Chat Settings", open=False):
                with gr.Row():
                    gr.HTML("<strong>Reasoning Method</strong>")
                    gr.HTML("<strong>Model</strong>")

                with gr.Row():
                    reasoning_type_values = [DEFAULT_SETTING] + self._app.default_settings.reasoning.settings["use"].choices
                    self.reasoning_types = gr.Dropdown(
                        choices=reasoning_type_values,
                        value=DEFAULT_SETTING,
                        container=False,
                        show_label=False,
                    )
                    self.model_types = gr.Dropdown(
                        choices=self._app.default_settings.reasoning.options["simple"].settings["llm"].choices,
                        value="",
                        container=False,
                        show_label=False,
                    )

    def extract_keywords(self, text: str, allowed_postags: List[str] = ['NOUN', 'PROPN', 'ADJ', 'VERB']) -> List[str]:
        """
        Extracts keywords from the input text based on part-of-speech tagging.
        """
        doc = self.nlp(text.lower())
        keywords = [
            token.text for token in doc
            if token.pos_ in allowed_postags
            and token.text not in STOP_WORDS
            and token.text not in punctuation
            and len(token.text) > 2
        ]
        # Remove duplicates while preserving order
        seen = set()
        return [x for x in keywords if not (x in seen or seen.add(x))]

    def extract_code_snippets(self, table: BeautifulSoup) -> List[str]:
        code_snippets = []
        for row in table.find_all('tr'):
            file_cell = row.find('td', class_='f')
            code_cell = row.find('code', class_='con')

            if not (file_cell and code_cell):
                continue

            file_link = file_cell.find('a')
            if file_link:
                file_name = file_link.get_text(strip=True)
                file_href = f"http://super.dl.net:9999" + file_link.get('href', '#')
                _, ext = os.path.splitext(file_name)
                ext = ext.lower()

                if ext in self.EXCLUDED_EXTENSIONS:
                    continue

                code_snippets.append(f"<p><strong>File:</strong> <a href=\"{file_href}\">{file_name}</a></p>")

            cleaned_code_lines = []
            for code_line in code_cell.find_all('a', class_='s'):
                line_number_tag = code_line.find('span', class_='l')
                if line_number_tag:
                    line_number_tag.decompose()

                code_text = ''.join(code_line.stripped_strings)
                cleaned_code_lines.append(code_text)

            if cleaned_code_lines:
                code_block = "\n".join(cleaned_code_lines)
                code_snippets.append(f"<pre><code class=\"language-java\">\n{code_block}\n</code></pre>")

        return code_snippets if code_snippets else ["<p>No relevant information found in search results.</p>"]

    def format_search_results(self, table: BeautifulSoup) -> str:
        formatted_html = ""
        current_directory = ""

        for row in table.find_all('tr'):
            row_classes = row.get('class', [])
            if 'dir' in row_classes:
                dir_link = row.find('a')
                if dir_link:
                    current_directory = dir_link.get_text(strip=True)
                    dir_href = dir_link.get('href', '#')
                    formatted_html += f"<h3>📂 <strong>Directory:</strong> <a href=\"{dir_href}\">{current_directory}</a></h3>\n"
                continue  # Skip to next row after processing directory

            file_cell = row.find('td', class_='f')
            code_cell = row.find('code', class_='con')

            if not (file_cell and code_cell):
                continue

            file_link = file_cell.find('a')
            if file_link:
                file_name = file_link.get_text(strip=True)
                file_href = file_link.get('href', '#')
                _, ext = os.path.splitext(file_name)
                ext = ext.lower()

                if ext in self.EXCLUDED_EXTENSIONS:
                    continue

                formatted_html += f"<p><strong>File:</strong> <a href=\"{file_href}\">{file_name}</a></p>\n"

            cleaned_code_lines = []
            for code_line in code_cell.find_all('a', class_='s'):
                line_number_tag = code_line.find('span', class_='l')
                if line_number_tag:
                    line_number_tag.decompose()

                code_text = ''.join(code_line.stripped_strings)
                cleaned_code_lines.append(code_text)

            if cleaned_code_lines:
                code_block = "\n".join(cleaned_code_lines)
                formatted_html += (
                    "<details>\n"
                    "<summary>📄 <strong>Code Snippets</strong></summary>\n"
                    f"<pre><code class=\"language-java\">\n{code_block}\n</code></pre>\n"
                    "</details>\n"
                )

        return formatted_html if formatted_html else "<p>No relevant information found in search results.</p>"

    def chat_fn(
        self,
        conversation_id: str,
        chat_history: List[List[str]],
        settings: dict,
        reasoning_type: Optional[str],
        llm_type: Optional[str],
        state: dict,
        user_id: int,
        enable_code_search: bool,
        search_projects: List[str],
        search_defs: str,
        search_refs: str,
        search_path: str,
        search_hist: str,
        search_type: str,
        *selecteds,
    ):
        """
        Chat function with keyword-based search and context inclusion using RAG.
        """

        self.stop_flags[conversation_id] = False
        chat_input = chat_history[-1][0]
        chat_history = chat_history[:-1]

        pipeline, reasoning_state = self.create_pipeline(settings, reasoning_type, llm_type, state, user_id, *selecteds)
        print("Reasoning state:", reasoning_state)
        pipeline.set_output_queue(asyncio.Queue())

        text, refs, plot, plot_gr = "", "", None, gr.update(visible=False)
        msg_placeholder = getattr(flowsettings, "KH_CHAT_MSG_PLACEHOLDER", "Thinking...")
        print(msg_placeholder)

        # Initial placeholder yield
        yield (
            chat_history + [(chat_input, text or msg_placeholder)],
            refs,
            plot_gr,
            plot,
            state,
        )

        additional_context = ""
        if enable_code_search:
            keywords = self.extract_keywords(chat_input)
            print(f"Extracted Keywords: {keywords}")

            if not keywords:
                search_message = (
                    "### ⚠️ **Search Warning**\n"
                    "No significant keywords were found in your query. Please refine your question to include more specific terms.\n\n"
                    "[🔗 **View Search Results**](#)"
                )
                text += f"\n\n{search_message}\n\n"
                yield (
                    chat_history + [(chat_input, text or msg_placeholder)],
                    refs,
                    plot_gr,
                    plot,
                    state,
                )
            else:
                query = " ".join(keywords)
                params = {
                    'project': search_projects or self.projects,
                    'defs': search_defs,
                    'refs': search_refs,
                    'path': search_path,
                    'hist': search_hist,
                    'type': search_type,
                    'full': query,
                    'nn': '2',
                    'si': 'full',
                }
                search_url = f"http://super.dl.net:9999/search?{urllib.parse.urlencode(params, doseq=True)}"
                print(f"Constructed Search URL: {search_url}")

                try:
                    response = requests.get(search_url)
                    content_type = response.headers.get('Content-Type', '')
                    if 'text/html' in content_type:
                        print("Received HTML response for search query.")
                        soup = BeautifulSoup(response.text, 'html.parser')
                        results_table = soup.find('table', {'aria-label': 'table of results'})
                        if results_table:
                            code_snippets = self.extract_code_snippets(results_table)
                            additional_context = "\n\n".join(code_snippets)

                            search_message = (
                                f"<h3>🛠️ <strong>Source Code Data</strong></h3>\n"
                                f"<p><a href=\"{search_url}\">📎 <strong>View Search Results</strong></a></p>\n"
                                f"{additional_context}"
                            )
                        else:
                            search_message = (
                                "<h3>⚠️ <strong>Search Warning</strong></h3>\n"
                                "<p>'<table aria-label=\"table of results\">' not found in the HTML response.</p>\n"
                                f"<p><a href=\"{search_url}\">🔗 <strong>View Search Results</strong></a></p>\n"
                            )
                    else:
                        search_message = (
                            "<h3>⚠️ <strong>Search Warning</strong></h3>\n"
                            "<p>Received an unexpected response type.</p>\n"
                            f"<p><a href=\"{search_url}\">🔗 <strong>View Search Results</strong></a></p>\n"
                        )
                except Exception as e:
                    search_message = (
                        "<h3>❌ <strong>Search Error</strong></h3>\n"
                        "<p>An unexpected error occurred during search.</p>\n"
                        f"<p><strong>Error Details:</strong> {str(e)}</p>\n"
                        f"<p><a href=\"{search_url}\">🔗 <strong>View Search Results</strong></a></p>\n"
                    )

                combined_prompt = f"User Query:\n{chat_input}\n\n{search_message}\n\n"

                # Convert markdown to HTML
                refs = search_message

                # Yield updated chat with combined prompt
                yield (
                    chat_history + [(chat_input, text or msg_placeholder)],
                    refs,
                    plot_gr,
                    plot,
                    state,
                )
        else:
            combined_prompt = chat_input

        # Model Response Handling
        for response in pipeline.stream(combined_prompt, conversation_id, chat_history):
            # Check if stop has been requested
            if self.stop_flags.get(conversation_id, False):
                print("Stop requested by user.")
                break  # Exit the response generation loop

            if not isinstance(response, Document) or response.channel is None:
                continue

            if response.channel == "chat":
                text += response.content or ""
            elif response.channel == "info":
                refs += response.content or ""
            elif response.channel == "plot":
                plot = response.content
                plot_gr = self._json_to_plot(plot)

            state[pipeline.get_info()["id"]] = reasoning_state["pipeline"]

            yield (
                chat_history + [(chat_input, text or msg_placeholder)],
                refs,
                plot_gr,
                plot,
                state,
            )

        # Finalizing response
        if not text:
            empty_msg = getattr(flowsettings, "KH_CHAT_EMPTY_MSG_PLACEHOLDER", "(Sorry, I don't know)")
            print(f"Generate nothing: {empty_msg}")
            text = empty_msg

        chat_history.append((chat_input, text))
        state[pipeline.get_info()["id"]] = reasoning_state["pipeline"]

        yield (
            chat_history,
            refs,
            plot_gr,
            plot,
            state,
        )

        # Clean up stop flag
        if conversation_id in self.stop_flags:
            del self.stop_flags[conversation_id]

    def _json_to_plot(self, json_dict: Optional[dict]):
        """
        Converts JSON to Plotly plot.
        """
        if json_dict:
            plot = from_json(json_dict)
            return gr.update(visible=True, value=plot)
        return gr.update(visible=False)

    def on_register_events(self):
        """
        Registers all necessary Gradio events.
        """
        self._register_chat_events()
        self._register_regen_events()
        self._register_ui_events()
        self._register_like_and_new_conv_events()
        self._register_delete_and_rename_events()
        self._register_conversation_selection_events()
        self._register_public_conversation_event()
        self._register_report_issue_event()
        self._register_setting_change_events()

        # Register Stop button event
        self.chat_panel.stop_btn.click(
            fn=self.handle_stop_request,
            inputs=self.chat_control.conversation_id,  # Pass conversation_id
            outputs=None,
            show_progress="hidden",
        )

    def handle_stop_request(self, conversation_id: str):
        """
        Sets the stop flag for the given conversation_id.
        """
        if conversation_id in self.stop_flags:
            self.stop_flags[conversation_id] = True
            print(f"Stop requested for conversation_id: {conversation_id}")
        else:
            print(f"No active conversation with id: {conversation_id}")
        return None

    def _register_chat_events(self):
        """Registers chat submission events."""
        chat_event = (
            gr.on(
                triggers=[self.chat_panel.text_input.submit, self.chat_panel.submit_btn.click],
                fn=self.submit_msg,
                inputs=[
                    self.chat_panel.text_input,
                    self.chat_panel.chatbot,
                    self._app.user_id,
                    self.chat_control.conversation_id,
                    self.chat_control.conversation_rn,
                    self.state_follow_up,
                ],
                outputs=[
                    self.chat_panel.text_input,
                    self.chat_panel.chatbot,
                    self.chat_control.conversation_id,
                    self.chat_control.conversation,
                    self.chat_control.conversation_rn,
                    self.state_follow_up,
                ],
                concurrency_limit=20,
                show_progress="hidden",
            )
            .success(
                fn=self.chat_fn,
                inputs=[
                    self.chat_control.conversation_id,
                    self.chat_panel.chatbot,
                    self._app.settings_state,
                    self._reasoning_type,
                    self._llm_type,
                    self.state_chat,
                    self._app.user_id,
                    self.enable_code_search,
                    self.search_projects,
                    self.search_defs,
                    self.search_refs,
                    self.search_path,
                    self.search_hist,
                    self.search_type,
                ] + self._indices_input,
                outputs=[
                    self.chat_panel.chatbot,
                    self.info_panel,
                    self.plot_panel,
                    self.state_plot_panel,
                    self.state_chat,
                ],
                concurrency_limit=20,
                show_progress="minimal",
            )
            .then(
                fn=lambda: True,
                inputs=None,
                outputs=[self._preview_links],
                js=PDFVIEW_JS,
            )
            .success(
                fn=self.check_and_suggest_name_conv,
                inputs=self.chat_panel.chatbot,
                outputs=[self.chat_control.conversation_rn, self._conversation_renamed],
            )
            .success(
                self.chat_control.rename_conv,
                inputs=[
                    self.chat_control.conversation_id,
                    self.chat_control.conversation_rn,
                    self._conversation_renamed,
                    self._app.user_id,
                ],
                outputs=[
                    self.chat_control.conversation,
                    self.chat_control.conversation,
                    self.chat_control.conversation_rn,
                ],
                show_progress="hidden",
            )
        )

        if getattr(flowsettings, "KH_FEATURE_CHAT_SUGGESTION", False):
            chat_event = (
                chat_event
                .success(
                    fn=self.suggest_chat_conv,
                    inputs=[self._app.settings_state, self.chat_panel.chatbot],
                    outputs=[self.state_follow_up, self._suggestion_updated],
                    show_progress="hidden",
                )
                .success(
                    self.chat_control.persist_chat_suggestions,
                    inputs=[
                        self.chat_control.conversation_id,
                        self.state_follow_up,
                        self._suggestion_updated,
                        self._app.user_id,
                    ],
                    outputs=None,
                    show_progress="hidden",
                )
            )

        chat_event.then(
            fn=self.persist_data_source,
            inputs=[
                self.chat_control.conversation_id,
                self._app.user_id,
                self.info_panel,
                self.state_plot_panel,
                self.state_retrieval_history,
                self.state_plot_history,
                self.chat_panel.chatbot,
                self.state_chat,
            ] + self._indices_input,
            outputs=[self.state_retrieval_history, self.state_plot_history],
            concurrency_limit=20,
        )

    def _register_regen_events(self):
        """Registers chat regeneration events."""
        regen_event = (
            self.chat_panel.regen_btn.click(
                fn=self.regen_fn,
                inputs=[
                    self.chat_control.conversation_id,
                    self.chat_panel.chatbot,
                    self._app.settings_state,
                    self._reasoning_type,
                    self._llm_type,
                    self.state_chat,
                    self._app.user_id,
                    self.enable_code_search,
                    self.search_projects,
                    self.search_defs,
                    self.search_refs,
                    self.search_path,
                    self.search_hist,
                    self.search_type,
                ] + self._indices_input,
                outputs=[
                    self.chat_panel.chatbot,
                    self.info_panel,
                    self.plot_panel,
                    self.state_plot_panel,
                    self.state_chat,
                ],
                concurrency_limit=20,
                show_progress="minimal",
            )
            .then(
                fn=lambda: True,
                inputs=None,
                outputs=[self._preview_links],
                js=PDFVIEW_JS,
            )
            .success(
                fn=self.check_and_suggest_name_conv,
                inputs=self.chat_panel.chatbot,
                outputs=[self.chat_control.conversation_rn, self._conversation_renamed],
            )
            .success(
                self.chat_control.rename_conv,
                inputs=[
                    self.chat_control.conversation_id,
                    self.chat_control.conversation_rn,
                    self._conversation_renamed,
                    self._app.user_id,
                ],
                outputs=[
                    self.chat_control.conversation,
                    self.chat_control.conversation,
                    self.chat_control.conversation_rn,
                ],
                show_progress="hidden",
            )
        )

        if getattr(flowsettings, "KH_FEATURE_CHAT_SUGGESTION", False):
            regen_event = (
                regen_event
                .success(
                    fn=self.suggest_chat_conv,
                    inputs=[self._app.settings_state, self.chat_panel.chatbot],
                    outputs=[self.state_follow_up, self._suggestion_updated],
                    show_progress="hidden",
                )
                .success(
                    self.chat_control.persist_chat_suggestions,
                    inputs=[
                        self.chat_control.conversation_id,
                        self.state_follow_up,
                        self._suggestion_updated,
                        self._app.user_id,
                    ],
                    outputs=None,
                    show_progress="hidden",
                )
            )

        regen_event.then(
            fn=self.persist_data_source,
            inputs=[
                self.chat_control.conversation_id,
                self._app.user_id,
                self.info_panel,
                self.state_plot_panel,
                self.state_retrieval_history,
                self.state_plot_history,
                self.chat_panel.chatbot,
                self.state_chat,
            ] + self._indices_input,
            outputs=[self.state_retrieval_history, self.state_plot_history],
            concurrency_limit=20,
        )

    def _register_ui_events(self):
        """Registers UI-related events."""
        self.chat_control.btn_info_expand.click(
            fn=lambda is_expanded: (
                gr.update(scale=INFO_PANEL_SCALES[is_expanded]),
                not is_expanded,
            ),
            inputs=self._info_panel_expanded,
            outputs=[self.info_column, self._info_panel_expanded],
        )

    def _register_like_and_new_conv_events(self):
        """Registers like and new conversation events."""
        self.chat_panel.chatbot.like(
            fn=self.is_liked,
            inputs=[self.chat_control.conversation_id],
            outputs=None,
        )
        self.chat_control.btn_new.click(
            self.chat_control.new_conv,
            inputs=self._app.user_id,
            outputs=[self.chat_control.conversation_id, self.chat_control.conversation],
            show_progress="hidden",
        ).then(
            self.chat_control.select_conv,
            inputs=[self.chat_control.conversation, self._app.user_id],
            outputs=[
                self.chat_control.conversation_id,
                self.chat_control.conversation,
                self.chat_control.conversation_rn,
                self.chat_panel.chatbot,
                self.state_follow_up,
                self.info_panel,
                self.state_plot_panel,
                self.state_retrieval_history,
                self.state_plot_history,
                self.chat_control.cb_is_public,
                self.state_chat,
            ] + self._indices_input,
            show_progress="hidden",
        ).then(
            fn=self._json_to_plot,
            inputs=self.state_plot_panel,
            outputs=self.plot_panel,
        )

    def _register_delete_and_rename_events(self):
        """Registers delete and rename conversation events."""
        self.chat_control.btn_del.click(
            fn=lambda id: self.toggle_delete(id),
            inputs=[self.chat_control.conversation_id],
            outputs=[self.chat_control._new_delete, self.chat_control._delete_confirm],
        )
        self.chat_control.btn_del_conf.click(
            self.chat_control.delete_conv,
            inputs=[self.chat_control.conversation_id, self._app.user_id],
            outputs=[self.chat_control.conversation_id, self.chat_control.conversation],
            show_progress="hidden",
        ).then(
            self.chat_control.select_conv,
            inputs=[self.chat_control.conversation, self._app.user_id],
            outputs=[
                self.chat_control.conversation_id,
                self.chat_control.conversation,
                self.chat_control.conversation_rn,
                self.chat_panel.chatbot,
                self.state_follow_up,
                self.info_panel,
                self.state_plot_panel,
                self.state_retrieval_history,
                self.state_plot_history,
                self.chat_control.cb_is_public,
                self.state_chat,
            ] + self._indices_input,
            show_progress="hidden",
        ).then(
            fn=self._json_to_plot,
            inputs=self.state_plot_panel,
            outputs=self.plot_panel,
        ).then(
            lambda: self.toggle_delete(""),
            outputs=[self.chat_control._new_delete, self.chat_control._delete_confirm],
        )
        self.chat_control.btn_del_cnl.click(
            fn=lambda: self.toggle_delete(""),
            outputs=[self.chat_control._new_delete, self.chat_control._delete_confirm],
        )
        self.chat_control.btn_conversation_rn.click(
            fn=lambda: gr.update(visible=True),
            outputs=[self.chat_control.conversation_rn],
        )
        self.chat_control.conversation_rn.submit(
            self.chat_control.rename_conv,
            inputs=[
                self.chat_control.conversation_id,
                self.chat_control.conversation_rn,
                gr.State(value=True),
                self._app.user_id,
            ],
            outputs=[
                self.chat_control.conversation,
                self.chat_control.conversation,
                self.chat_control.conversation_rn,
            ],
            show_progress="hidden",
        )

    def _register_conversation_selection_events(self):
        """Registers conversation selection events."""
        self.chat_control.conversation.select(
            self.chat_control.select_conv,
            inputs=[self.chat_control.conversation, self._app.user_id],
            outputs=[
                self.chat_control.conversation_id,
                self.chat_control.conversation,
                self.chat_control.conversation_rn,
                self.chat_panel.chatbot,
                self.state_follow_up,
                self.info_panel,
                self.state_plot_panel,
                self.state_retrieval_history,
                self.state_plot_history,
                self.chat_control.cb_is_public,
                self.state_chat,
            ] + self._indices_input,
            show_progress="hidden",
        ).then(
            fn=self._json_to_plot,
            inputs=self.state_plot_panel,
            outputs=self.plot_panel,
        ).then(
            lambda: self.toggle_delete(""),
            outputs=[self.chat_control._new_delete, self.chat_control._delete_confirm],
        ).then(
            fn=None, inputs=None, outputs=None, js=PDFVIEW_JS
        )

    def _register_public_conversation_event(self):
        """Registers public conversation toggle event."""
        self.chat_control.cb_is_public.change(
            self.on_set_public_conversation,
            inputs=[self.chat_control.cb_is_public, self.chat_control.conversation],
            outputs=None,
            show_progress="hidden",
        )

    def _register_report_issue_event(self):
        """Registers report issue event."""
        self.report_issue.report_btn.click(
            self.report_issue.report,
            inputs=[
                self.report_issue.correctness,
                self.report_issue.issues,
                self.report_issue.more_detail,
                self.chat_control.conversation_id,
                self.chat_panel.chatbot,
                self._app.settings_state,
                self._app.user_id,
                self.info_panel,
                self.state_chat,
            ] + self._indices_input,
            outputs=None,
        )

    def _register_setting_change_events(self):
        """Registers reasoning and model type change events."""
        self.reasoning_types.change(
            self.reasoning_changed,
            inputs=[self.reasoning_types],
            outputs=[self._reasoning_type],
        )
        self.model_types.change(
            lambda x: x,
            inputs=[self.model_types],
            outputs=[self._llm_type],
        )

        if getattr(flowsettings, "KH_FEATURE_CHAT_SUGGESTION", False):
            self.state_follow_up.select(
                self.chat_control.chat_suggestion.select_example,
                outputs=[self.chat_panel.text_input],
                show_progress="hidden",
            )

    def submit_msg(
        self, chat_input, chat_history, user_id, conv_id, conv_name, chat_suggest
    ):
        """
        Submit a message to the chatbot.
        """
        if not chat_input:
            raise ValueError("Input is empty")

        chat_input_text = chat_input.get("text", "")

        # check if regen mode is active
        if chat_input_text:
            chat_history = chat_history + [(chat_input_text, None)]
        else:
            if not chat_history:
                raise gr.Error("Empty chat")

        if not conv_id:
            conv_id, conv_update = self.chat_control.new_conv(user_id)
            with Session(engine) as session:
                statement = select(Conversation).where(Conversation.id == conv_id)
                conversation = session.exec(statement).one()
                new_conv_name = conversation.name
                new_chat_suggestion = conversation.data_source.get("chat_suggestions", [])
        else:
            conv_update = gr.update()
            new_conv_name = conv_name
            new_chat_suggestion = chat_suggest

        return (
            {},
            chat_history,
            conv_id,
            conv_update,
            new_conv_name,
            new_chat_suggestion,
        )

    def toggle_delete(self, conv_id: str):
        """
        Toggles the delete confirmation visibility.
        """
        return (
            gr.update(visible=not bool(conv_id)),
            gr.update(visible=bool(conv_id)),
        )

    def on_set_public_conversation(self, is_public: bool, convo_id: str):
        """
        Sets the conversation as public or private.
        """
        if not convo_id:
            gr.Warning("No conversation selected")
            return

        with Session(engine) as session:
            statement = select(Conversation).where(Conversation.id == convo_id)
            result = session.exec(statement).one()
            name = result.name

            if result.is_public != is_public:
                result.is_public = is_public
                session.add(result)
                session.commit()
                gr.Info(f"Conversation: {name} is {'public' if is_public else 'private'}.")

    def on_subscribe_public_events(self):
        """
        Subscribes to public events like sign-in and sign-out.
        """
        if self._app.f_user_management:
            self._app.subscribe_event(
                name="onSignIn",
                definition={
                    "fn": self.chat_control.reload_conv,
                    "inputs": [self._app.user_id],
                    "outputs": [self.chat_control.conversation],
                    "show_progress": "hidden",
                },
            )

            self._app.subscribe_event(
                name="onSignOut",
                definition={
                    "fn": lambda: self.chat_control.select_conv("", None),
                    "outputs": [
                        self.chat_control.conversation_id,
                        self.chat_control.conversation,
                        self.chat_control.conversation_rn,
                        self.chat_panel.chatbot,
                        self.info_panel,
                        self.state_plot_panel,
                        self.state_retrieval_history,
                        self.state_plot_history,
                        self.chat_control.cb_is_public,
                    ] + self._indices_input,
                    "show_progress": "hidden",
                },
            )

    def persist_data_source(
        self,
        convo_id: str,
        user_id: int,
        retrieval_msg: str,
        plot_data: dict,
        retrival_history: List[str],
        plot_history: List[dict],
        messages: List[List[str]],
        state: dict,
        *selecteds,
    ):
        """
        Updates the data source with the latest conversation data.
        """
        if not convo_id:
            gr.Warning("No conversation selected")
            return

        if not state["app"].get("regen", False):
            retrival_history.append(retrieval_msg)
            plot_history.append(plot_data)
        else:
            if retrival_history:
                print("Updating retrieval history (regen=True)")
                retrival_history[-1] = retrieval_msg
                plot_history[-1] = plot_data

        state["app"]["regen"] = False

        selecteds_ = {}
        for index in self._app.index_manager.indices:
            if index.selector is None:
                continue
            if isinstance(index.selector, int):
                selecteds_[str(index.id)] = selecteds[index.selector]
            else:
                selecteds_[str(index.id)] = [selecteds[i] for i in index.selector]

        with Session(engine) as session:
            statement = select(Conversation).where(Conversation.id == convo_id)
            result = session.exec(statement).one()

            data_source = result.data_source
            old_selecteds = data_source.get("selected", {})
            is_owner = result.user == user_id

            result.data_source = {
                "selected": selecteds_ if is_owner else old_selecteds,
                "messages": messages,
                "retrieval_messages": retrival_history,
                "plot_history": plot_history,
                "state": state,
                "likes": deepcopy(data_source.get("likes", [])),
            }
            session.add(result)
            session.commit()

        return retrival_history, plot_history

    def reasoning_changed(self, reasoning_type: str):
        """
        Handles changes in reasoning type.
        """
        if reasoning_type != DEFAULT_SETTING:
            gr.Info(f"Reasoning type changed to `{reasoning_type}`")
        return reasoning_type

    def is_liked(self, convo_id: str, liked: gr.LikeData):
        """
        Handles like interactions on conversations.
        """
        with Session(engine) as session:
            statement = select(Conversation).where(Conversation.id == convo_id)
            result = session.exec(statement).one()

            data_source = deepcopy(result.data_source)
            likes = data_source.get("likes", [])
            likes.append([liked.index, liked.value, liked.liked])
            data_source["likes"] = likes

            result.data_source = data_source
            session.add(result)
            session.commit()

    def message_selected(self, retrieval_history: List[str], plot_history: List[dict], msg: gr.SelectData):
        """
        Handles selection of a message from retrieval history.
        """
        index = msg.index[0]
        try:
            retrieval_content = retrieval_history[index]
            plot_content = plot_history[index]
        except IndexError:
            retrieval_content, plot_content = gr.update(), None

        return retrieval_content, plot_content

    def create_pipeline(
        self,
        settings: dict,
        session_reasoning_type: Optional[str],
        session_llm: Optional[str],
        state: dict,
        user_id: int,
        *selecteds,
    ):
        """
        Creates the pipeline from settings.
        """
        print("Session reasoning type:", session_reasoning_type)
        print("Session LLM:", session_llm)
        reasoning_mode = settings["reasoning.use"] if session_reasoning_type in (DEFAULT_SETTING, None) else session_reasoning_type
        reasoning_cls = reasonings[reasoning_mode]
        print("Reasoning class:", reasoning_cls)
        reasoning_id = reasoning_cls.get_info()["id"]

        settings = deepcopy(settings)
        llm_setting_key = f"reasoning.options.{reasoning_id}.llm"
        if llm_setting_key in settings and session_llm not in (DEFAULT_SETTING, None):
            settings[llm_setting_key] = session_llm

        retrievers = []
        for index in self._app.index_manager.indices:
            index_selected = []
            if isinstance(index.selector, int):
                index_selected = selecteds[index.selector]
            elif isinstance(index.selector, tuple):
                index_selected = [selecteds[i] for i in index.selector]
            retrievers += index.get_retriever_pipelines(settings, user_id, index_selected)

        reasoning_state = {
            "app": deepcopy(state["app"]),
            "pipeline": deepcopy(state.get(reasoning_id, {})),
        }

        pipeline = reasoning_cls.get_pipeline(settings, reasoning_state, retrievers)
        return pipeline, reasoning_state

    def regen_fn(
        self,
        conversation_id: str,
        chat_history: List[List[str]],
        settings: dict,
        reasoning_type: Optional[str],
        llm_type: Optional[str],
        state: dict,
        user_id: int,
        enable_code_search: bool,
        search_projects: List[str],
        search_defs: str,
        search_refs: str,
        search_path: str,
        search_hist: str,
        search_type: str,
        *selecteds,
    ):
        """
        Regeneration function to replay the chat_fn with updated context.
        """
        if not chat_history:
            gr.Warning("Empty chat")
            yield chat_history, "", state
            return

        state["app"]["regen"] = True
        yield from self.chat_fn(
            conversation_id,
            chat_history,
            settings,
            reasoning_type,
            llm_type,
            state,
            user_id,
            enable_code_search,
            search_projects,
            search_defs,
            search_refs,
            search_path,
            search_hist,
            search_type,
            *selecteds,
        )

    def check_and_suggest_name_conv(self, chat_history: List[List[str]]):
        """
        Suggests a conversation name based on chat history.
        """
        suggest_pipeline = SuggestConvNamePipeline()
        new_name = gr.update()
        renamed = False

        if len(chat_history) == 1:
            suggested_name = suggest_pipeline(chat_history).text
            suggested_name = suggested_name.replace('"', "").replace("'", "")[:40]
            new_name = gr.update(value=suggested_name)
            renamed = True

        return new_name, renamed

    def suggest_chat_conv(self, settings: dict, chat_history: List[List[str]]):
        """
        Suggests follow-up questions based on chat history.
        """
        suggest_pipeline = SuggestFollowupQuesPipeline()
        suggest_pipeline.lang = SUPPORTED_LANGUAGE_MAP.get(settings.get("reasoning.lang"), "English")

        updated = False
        suggested_ques = []
        if len(chat_history) >= 1:
            suggested_resp = suggest_pipeline(chat_history).text
            ques_res = re.search(r"\[(.*?)\]", re.sub("\n", "", suggested_resp))
            if ques_res:
                try:
                    suggested_ques = [[x] for x in re.findall(r'\[(.*?)\]', ques_res.group())]
                    updated = True
                except Exception:
                    pass

        return suggested_ques, updated
