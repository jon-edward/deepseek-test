"""
A test of DeepSeek using torch.

This is a simple prompt-response loop that maintains a conversation history in memory.
"""

from typing import Tuple

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.styles import Style
from prompt_toolkit.validation import Validator, ValidationError

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.text import Text
from rich.theme import Theme


prompt_session = PromptSession()
user_prompt_style = Style.from_dict(
    {
        "user-prompt": "fg:#62ff42 bold",
    }
)


def split_think(s: str) -> Tuple[str, str]:
    """
    Split a string into two parts, one inside the <think> tags and one after the </think> tag.
    """
    start_tag = "<think>"
    end_tag = "</think>"

    try:
        start_think = s.index(start_tag)
    except ValueError:
        start_think = None

    try:
        end_think = s.index(end_tag)
    except ValueError:
        end_think = None

    if start_think is end_think is None:
        # There are no <think> or </think> tags, treat like a normal message.
        return "", s.strip()
    if start_think is None and end_think is not None:
        # There is only a </think> tag, split.
        return s[:end_think].strip(), s[end_think + len(end_tag) :].strip()
    if start_think is not None and end_think is None:
        # There is only a <think> tag, split.
        return s[start_think + len(start_tag) :].strip(), ""
    if start_think > end_think:
        # The </think> tag is before the <think> tag, treat like a normal message.
        return "", s.strip()

    # There is both a <think> and a </think> tag, split.
    return (
        s[start_think + len(start_tag) : end_think].strip(),
        s[end_think + len(end_tag) :].strip(),
    )


class UserPromptValidator(Validator):
    def validate(self, document):
        if not document.text.strip():
            raise ValidationError(message="Please enter a message.")


def user_prompt() -> str:
    return prompt_session.prompt(
        [("class:user-prompt", "\nYou:\n")],
        style=user_prompt_style,
        validator=UserPromptValidator(),
        auto_suggest=AutoSuggestFromHistory(),
        enable_history_search=True,
    )


if __name__ == "__main__":
    from chat import DeepSeekChat

    deepseek_chat = DeepSeekChat()
    console = Console()

    while True:
        user_message = user_prompt()
        deepseek_chat.add_user_message(user_message)

        console.print(f"\nDeepSeek:", style="bold blue")

        placeholder = Text("Generating response...")
        placeholder.stylize("grey62")

        with Live(placeholder, console=console) as live:
            # Iteratively generate responses from the chat history, updating the live console as we go
            # with Markdown formatting.
            result = ""

            for response in deepseek_chat.generate():
                result += response

                think, parsed_result = split_think(result)

                group_elements = []

                if think:
                    group_elements.append(
                        Padding(Markdown(think, style="grey62"), (0, 4))
                    )
                    group_elements.append(Text(""))
                if parsed_result:
                    group_elements.append(Markdown(parsed_result))
                    group_elements.append(Text(""))

                if group_elements:
                    group_elements.pop()

                markdown_group = Group(*group_elements)
                live.update(markdown_group)
