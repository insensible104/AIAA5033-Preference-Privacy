import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast


@pytest.fixture
def tokenizer():
    backend = Tokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "[EOS]": 2,
                "<user>": 3,
                "<assistant>": 4,
                "Hello": 5,
                "Hi": 6,
                "Question": 7,
                "Good": 8,
                "Bad": 9,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = WhitespaceSplit()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tok.chat_template = "{% for message in messages %}{{ '<' + message['role'] + '> ' + message['content'] + ' [EOS] ' }}{% endfor %}{% if add_generation_prompt %}{{ '<assistant> ' }}{% endif %}"
    return tok


@pytest.fixture
def row():
    return {
        "id": "sample",
        "prompt_id": "prompt",
        "prompt": "\n\nHuman: Hello\n\nAssistant: Hi\n\nHuman: Question\n\nAssistant:",
        "response_a": " Bad",
        "response_b": " Good",
        "label": 1,
    }
