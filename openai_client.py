import json
import time
from typing import Type, List

from dotenv import load_dotenv
from openai import OpenAI, APIError, RateLimitError
from pydantic import BaseModel, ValidationError

from utils import get_logger

client_logger = get_logger(name= "openai_client", filename= "openai_client.log", level= "debug")


class LLM:
    """
    OpenAI structured-output wrapper with intelligent retries.

    Features:
    - Strict Pydantic validation
    - Automatic retries
    - Retry prompt augmentation
    - Schema injection on retry
    - Helpful validation feedback
    """

    session_id = 0
    input_token = 0
    cached_token = 0
    output_token = 0

    def __init__(self,max_retries: int = 3, retry_delay: float = 0.5, sid_prefix:str=""):
        """
        Args:
            model:
                OpenAI model name
                Example:
                    "gpt-5-mini"

            max_retries:
                Maximum retry attempts

            retry_delay:
                Delay between retries in seconds
        """

        load_dotenv()

        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.sid_prefix = sid_prefix
        # Automatically reads OPENAI_API_KEY
        self.client = OpenAI()
    def _session_id(self):
        return self.sid_prefix +"-"+ str(self.session_id)
    def _build_retry_prompt(original_instructions: str,response_format: Type[BaseModel], error_message: str) -> str:
        """
        Augments the prompt with schema + correction instructions.
        """
        schema = json.dumps(response_format.model_json_schema(),indent=4)

        retry_prompt = f"""
                            {original_instructions}

                            IMPORTANT:
                            Your previous response failed validation.

                            Validation Error:
                            {error_message}

                            You MUST follow the exact JSON schema below.

                            STRICT REQUIREMENTS:
                            - Return ONLY valid JSON
                            - Do NOT include markdown
                            - Do NOT include explanations
                            - Do NOT omit required fields
                            - Use correct field types
                            - Ensure the response fully matches the schema

                            EXPECTED JSON SCHEMA:
                            {schema}
                            """

        return retry_prompt

    def generate_response(
        self,
        instructions: str,
        input_text: str,
        model_name: str,
        response_format: Type[BaseModel],
        version,
        reasoning
        ) -> dict:
        """
        Generates validated structured output.

        Automatically retries if:
        - JSON is malformed
        - Required fields are missing
        - Types are incorrect
        - Schema validation fails
        - Extra Fields
        """

        last_exception = None
        self.input_token = 0
        self.cached_token = 0
        self.output_token = 0

        current_instructions = instructions
        if version == "classifier":
            self.session_id += 1 

        for attempt in range(1, self.max_retries + 1):

            try:
                response = self.client.responses.parse(
                    model= model_name,
                    instructions=current_instructions,
                    input=input_text,
                    text_format=response_format,
                    reasoning= reasoning
                )

                parsed_output = response.output_parsed

                usage = response.usage
                self.input_token += usage.input_tokens
                self.cached_token += usage.input_tokens_details.cached_tokens
                self.output_token += usage.output_tokens

                # Safety check
                if parsed_output is None:
                    raise ValueError("The model returned empty structured output for %s Session ID: %s",version, self._session_id())
                return parsed_output.model_dump()

            except (
                ValidationError,
                ValueError,
                APIError,
                RateLimitError
            ) as e:

                last_exception = e

                # Stop if no retries remain
                if attempt >= self.max_retries:
                    break

                if isinstance(e, (ValidationError, ValueError)):
                    client_logger.error(f"|<{version}> Session Id: {self._session_id()} | Retry: {attempt}/{self.max_retries} | Validation failed \n{str(e)}\n retrying in {self.retry_delay} sec . . . . ")
                    # Build stronger retry prompt
                    current_instructions = self._build_retry_prompt(
                        original_instructions=instructions,
                        response_format=response_format,
                        error_message=str(e)
                    )

                    time.sleep(self.retry_delay)

                elif isinstance(e, APIError):

                    client_logger.error(f"|<{version}> Session Id: {self._session_id()} | Retry: {attempt}/{self.max_retries} | API failed \n{str(e)}\n retrying in {self.retry_delay*3} sec . . . . ")
                    time.sleep(self.retry_delay*3)

                elif isinstance(e, RateLimitError): 
                    client_logger.error(f"|<{version}> Session Id: {self._session_id()} | Retry: {attempt}/{self.max_retries} | Rate Limited!! \n{str(e)}\n , retrying in {self.retry_delay*5} sec . . . .")
                    time.sleep(self.retry_delay*5)

        client_logger.critical(f"| << {version} failed >>  | Session Id: {self._session_id()}|  Failed after retries: {self.max_retries}  \n{str(last_exception)}\n")
        raise TimeoutError(f"Session Id: {self._session_id()} {version} failed after retries: {self.max_retries} \n{str(last_exception)}\n")





class EmbeddingModel:
    """
    OpenAI embedding wrapper.

    Supports:
    - Single embedding generation
    - Batch embedding generation
    - Dynamic OpenAI embedding models
    """
    tokens = 0
    batch_tokens = 0
    def __init__(self, model_name: str):
        """
        Initializes the embedding model.

        Args:
            model_name:
                OpenAI embedding model name

                Examples:
                - "text-embedding-3-small"
                - "text-embedding-3-large"
        """

        load_dotenv()

        self.model_name = model_name

        # Automatically reads OPENAI_API_KEY
        self.client = OpenAI()

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generates a vector embedding for a single string.

        Args:
            text:
                Input text

        Returns:
            List[float]:
                Embedding vector
        """
        self.tokens = 0

        response = self.client.embeddings.create(
            model=self.model_name,
            input=text
        )
        self.tokens = response.usage.total_tokens
        return response.data[0].embedding

    def generate_batch_embeddings(
        self,
        texts: List[str]
    ) -> List[List[float]]:
        """
        Generates embeddings for multiple strings.

        Args:
            texts:
                List of input strings

        Returns:
            List[List[float]]:
                List of embedding vectors
                in the same order as inputs
        """
        self.batch_tokens = 0

        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )


        embeddings = [
            item.embedding
            for item in response.data
        ]
        self.batch_tokens = response.usage.total_tokens
        return embeddings