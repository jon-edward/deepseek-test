"""
A high level implementation of chat functionality for DeepSeek.
"""

import dataclasses
from typing import Iterable, TypedDict, Literal, Generator, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from template import template


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


class Message(TypedDict):
    """
    A message in the chat history.
    """

    role: Literal["user", "assistant"]
    content: str


@dataclasses.dataclass
class DeepSeekChat:
    """
    A high level implementation of chat functionality for DeepSeek.
    """

    temperature: float = 0.7

    max_tokens_per_generation: int = 256
    max_generations_per_prompt: int = 30

    recall_messages: int = 15
    model_ident: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    chat: list[Message] = dataclasses.field(default_factory=list, init=False)

    # Lazy properties
    _model: AutoModelForCausalLM | None = None
    _tokenizer: AutoTokenizer | None = None

    @property
    def recalled_chat(self) -> Iterable[Message]:
        """
        Get the chat history that can be recalled by the model.
        
        This also removes any </think> tags from all but the last assistant message.
        """

        recall_messages = self.recall_messages
        if self.recall_messages is None:
            recall_messages = len(self.chat)

        messages = [
            {"role": message["role"], "content": (
                    message["content"].split("</think>", 1)[-1]
                    if message["role"] == "assistant"
                    else message["content"]
                ),
            }
            for message in self.chat[-recall_messages:-1]
        ]

        messages.append(self.chat[-1])
        return messages

    @property
    def model(self) -> AutoModelForCausalLM:
        """
        Get the model. Creates it if it doesn't exist.
        """

        if self._model is None:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_ident,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(device)
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        """
        Get the tokenizer for the model. Creates it if it doesn't exist.
        """

        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_ident)
        return self._tokenizer

    def assistant_seed(self) -> Message:
        """
        Get the seed message for the assistant to generate from.
        """

        return {"role": "assistant", "content": ""}

    def feed(self, message: Message, force_add: bool = False):
        """
        Add to the chat history. If the last message has the same role, append to it. Otherwise, add a new message.
        """

        if (
            not force_add
            and len(self.chat) > 0
            and self.chat[-1]["role"] == message["role"]
        ):
            self.chat[-1]["content"] = f"{self.chat[-1]['content']}{message['content']}"
        else:
            self.chat.append(message)

    def add_user_message(self, content: str):
        """
        Add a user message to the chat history.
        """

        self.feed({"role": "user", "content": content})

    def _generate(self, continued: bool) -> Tuple[bool, str]:
        """
        Perform a single generation. Returns whether the generation should continue and the generated text.
        """

        outputs = self.tokenizer.apply_chat_template(
            self.recalled_chat,
            add_generation_prompt=not continued,
            continue_final_message=continued,
            return_dict=True,
            return_tensors="pt",
            chat_template=template,
        )

        inference = self.model.generate(
            **outputs,
            temperature=self.temperature,
            max_new_tokens=self.max_tokens_per_generation,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
        )

        response = self.tokenizer.decode(
            inference[0][len(outputs["input_ids"][0]) :], skip_special_tokens=True
        )

        if response.endswith(self.tokenizer.eos_token):
            response = response[: -len(self.tokenizer.eos_token)]
            self.feed({"role": "assistant", "content": response})
            return False, response

        self.feed({"role": "assistant", "content": response})
        return True, response

    def generate(self) -> Generator[str, None, None]:
        """
        Generate a response from the chat history. Yields the response as it is generated.
        """

        with torch.no_grad():
            self.feed(self.assistant_seed(), force_add=True)

            generation_count = 0
            should_continue = True

            while (
                should_continue and generation_count < self.max_generations_per_prompt
            ):
                should_continue, text = self._generate(generation_count > 0)
                generation_count += 1
                yield text

        return self.chat[-1]["content"]
