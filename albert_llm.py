import json
import re
from collections.abc import Iterator

import requests
from requests import Timeout

import langchain
from langchain.schema.language_model import LanguageModelInput
# Import identique à nim_llm.py
from langchain_core.messages import AIMessage as CoreAIMessage
from langchain_core.messages import BaseMessage

from danswer.configs.model_configs import GEN_AI_API_ENDPOINT
from danswer.configs.model_configs import GEN_AI_MAX_OUTPUT_TOKENS
from danswer.llm.interfaces import LLM
from danswer.llm.interfaces import LLMConfig
from danswer.llm.interfaces import ToolChoiceOptions
from danswer.llm.utils import convert_lm_input_to_basic_string
from danswer.utils.logger import setup_logger

# S'il y a un AIMessage dans langchain, l'importer :
from langchain.schema.messages import AIMessage
from langchain.schema.messages import HumanMessage
from langchain.schema.messages import SystemMessage

logger = setup_logger()


class ALBERTModelServer(LLM):
    """Class to use the Albert API (https://albert.api.etalab.gouv.fr/v1)
    in a manner similar to nim_llm.py.

    Adapt as necessary according to the actual specification of the Albert API.
    """

    @property
    def requires_api_key(self) -> bool:
        return True

    def __init__(
        self,
        model_provider: str,
        model_name: str,
        api_key: str | None,
        timeout: int,
        temperature: float = 0.0,
        endpoint: str | None = GEN_AI_API_ENDPOINT,
        max_output_tokens: int = GEN_AI_MAX_OUTPUT_TOKENS,
        custom_llm_provider: str = 'albert'
    ):
        # By default, set Albert's endpoint if none is provided
        if not endpoint:
            endpoint = "https://albert.api.etalab.gouv.fr"
            print(f"Setting default endpoint to: {endpoint}")

        # Assume the Albert API has an endpoint /v1/chat/completions
        # or /v1/completions. Adapt according to the actual documentation.
        self._temperature = temperature
        self._api_key = api_key
        # Example with /v1/chat/completions
        self._endpoint = endpoint + "/v1/chat/completions"
        self._max_output_tokens = max_output_tokens
        self._timeout = timeout
        self._custom_llm_provider = custom_llm_provider
        self._model_provider = model_provider
        self._model_name = model_name.split('/')[1] if '/' in model_name else model_name

    def _format_message(self, message: str) -> list[dict]:
        """
        Converts the "Danswer" message style (with 'System:', 'Human:')
        into a list of messages in an OpenAI-like format.

        Verify if Albert expects a similar or different format.
        """
        lines = message.split('\n')

        formatted_messages = []
        role = None
        content = ''
        for line in lines:
            if line.startswith('System:'):
                # New block => push the old one if present
                if role:
                    formatted_messages.append({"role": role, "content": content.strip()})
                    content = ''
                role = 'system'
                content += line[len('System: '):] + '\n'
            elif line.startswith('Human:'):
                if role:
                    formatted_messages.append({"role": role, "content": content.strip()})
                    content = ''
                role = 'user'
                content += line[len('Human: '):] + '\n'
            # Example adjustment for a "Do not respond" message
            elif line == 'Do not respond':
                role = 'user'
                content += line + '\n'
            else:
                content += line + '\n'

        # Add the last block if not empty
        if role:
            formatted_messages.append({"role": role, "content": content.strip()})
        return formatted_messages

    def _debug_msg(self, messages: list) -> None:
        """
        For debugging, prints the type and content of the messages.
        """
        logger.info(f'Langchain version: {langchain.__version__}')
        for i, msg in enumerate(messages):
            logger.info(f'Message {i+1}:')
            logger.info(f'\t{msg.__class__}')
            logger.info(f'\t{msg.__class__.__name__}')
            logger.info(f'\t{msg.__dir__()}')

    def _execute(self, input: LanguageModelInput) -> AIMessage:
        """
        Performs the HTTP call to the Albert API based on the formatted messages.
        """
        headers = {
            "Content-Type": "application/json",
            # Modify if Albert does not require an "Authorization: Bearer <token>" header
            "Authorization": f"Bearer {self._api_key}" if self._api_key else "",
        }

        # Convert to basic string
        ds_message = convert_lm_input_to_basic_string(input)
        # Example: data for an OpenAI-like endpoint
        data = {
            "model": f"{self._model_provider}/{self._model_name}",
            "messages": self._format_message(ds_message),
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
            # top_p, etc. to adjust if necessary
            "top_p": 1,
            # If the API supports streaming, you can try "stream": True
            "stream": False
        }

        try:
            response = requests.post(
                self._endpoint, headers=headers, json=data, timeout=self._timeout
            )
        except Timeout as error:
            raise Timeout(f"Model inference to {self._endpoint} timed out") from error

        # Log for debugging
        logger.info(f"ALBERT MSG: {ds_message}")
        logger.info(f"REQUEST DATA: {data}")
        logger.info(f"RESPONSE STATUS CODE: {response.status_code}")

        response.raise_for_status()  # raises an exception if HTTP code != 2XX

        # Assume the response is similar to OpenAI ChatCompletion
        # and we get a JSON with "choices" => [ { "message": { "content": "..." } } ]
        response_json = response.json()
        response_content = response_json.get("choices", [])[0].get("message", {}).get("content", "")

        # If needed, filter specific content:
        # if self._model_provider == "XXX" and self._model_name == "YYY":
        #     response_content = re.sub(r"<pattern_to_remove>", "", response_content, flags=re.DOTALL).strip()

        # Return an AIMessage
        return AIMessage(content=response_content)

    @property
    def config(self) -> LLMConfig:
        """
        Returns the LLM configuration for internal use.
        """
        return LLMConfig(
            model_provider=self._model_provider,
            model_name=self._model_name,
            temperature=self._temperature,
            api_key=self._api_key,
        )

    def log_model_configs(self) -> None:
        logger.debug(f"Albert LLM at: {self._endpoint}")

    def invoke(
        self,
        prompt: LanguageModelInput,
        tools: list[dict] | None = None,
        tool_choice: ToolChoiceOptions | None = None,
    ) -> BaseMessage:
        """
        Main method used by Danswer to obtain a response.
        """
        return self._execute(prompt)

    def stream(
        self,
        prompt: LanguageModelInput,
        tools: list[dict] | None = None,
        tool_choice: ToolChoiceOptions | None = None,
    ) -> Iterator[BaseMessage]:
        """
        If the API supports streaming, we can iterate over the response.
        Here, we simply return a single block of response.
        """
        yield self._execute(prompt)
