from agent.models import ollama_model
from agent.models import openai_model
from agent.models import deepseek_model
from utils.image import encode_image


# pytest test_models.py::test_openai_model -s
def test_openai_model():
    response = openai_model.invoke("你是谁？请自我介绍一下。")
    assert response is not None
    print(response)


# pytest test_models.py::test_ollama_model -s
def test_ollama_model():
    response = ollama_model.invoke("你是谁？请自我介绍一下。")
    assert response is not None
    print(response)


# pytest test_models.py::test_ollama_model_image_message -s
def test_ollama_model_image_message():
    base64_image = encode_image(
        "/Users/yebaoping/code/python/AssistantBot/resources/cat.jpg"
    )

    response = ollama_model.invoke(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "这张图片里有什么？"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ]
    )
    assert response is not None
    print(response)

# pytest test_models.py::test_deepseek_model -s
def test_deepseek_model():
    response = deepseek_model.invoke("你是谁？请自我介绍一下。")
    assert response is not None
    print(response)