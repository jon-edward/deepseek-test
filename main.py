"""
A test of Deepseek using torch.

This is a simple prompt-response loop that maintains a conversation history in memory.
"""

from textwrap import indent
from typing import Tuple

from prompt_toolkit import prompt, PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.key_binding.bindings.basic import load_basic_bindings
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress


session = PromptSession()
style = Style.from_dict(
    {
        "user-prompt": "green bold",
    }
)


def split_think(s: str) -> Tuple[str, str]:
    s1, *s2 = s.split("</think>", 1)
    if not s2 or not s2[0]:
        return "", s1.strip()
    return s1.replace("<think>", "").strip(), s2[0].strip()


def user_prompt() -> str:
    return session.prompt(
        [("class:user-prompt", "You:\n")],
        style=style,
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

        print()
        result = ""

        with Progress(transient=True) as progress:
            progress.add_task("Generating...", total=None)
            for response in deepseek_chat.generate():
                result += response

        think, result = split_think(result)
        if think:
            result = f"{indent(think, ">")}\n\n{result}\n"

        console.print("DeepSeek: ", style="bold blue")
        console.print(Markdown(result))
        print()
