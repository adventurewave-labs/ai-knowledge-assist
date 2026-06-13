import pytest

from app.rag.prompts import get_system_prompt

EXPECTED_PROMPTS = ["default", "technical", "strict"]


@pytest.mark.parametrize("name", EXPECTED_PROMPTS)
def test_prompt_loads(name):
    prompt = get_system_prompt(name)
    assert isinstance(prompt, str)
    assert prompt.strip()


@pytest.mark.parametrize("name", EXPECTED_PROMPTS)
def test_prompt_contains_context_placeholder(name):
    prompt = get_system_prompt(name)
    assert "{context}" in prompt


def test_default_prompt_loads():
    assert get_system_prompt().strip()


def test_missing_prompt_raises_key_error():
    with pytest.raises(KeyError):
        get_system_prompt("does_not_exist")
