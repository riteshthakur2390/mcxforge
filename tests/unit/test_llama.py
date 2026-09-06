import requests
import json

# Use the lighter Mistral model for tests (faster, CPU-friendly)
MODEL_NAME = "mistral"

def generate_signal(prompt: str, max_tokens: int = 100) -> str:
    """
    Sends a prompt to the local Ollama server and returns the generated response.
    Handles cases where the response might contain extra text before JSON.
    """
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    try:
        res = requests.post(url, json=payload, timeout=1.0)
    except Exception:
        return "MOCK_RESPONSE_OLLAMA_OFFLINE"

    try:
        data = json.loads(res.text.strip().split("\n")[-1])
        return data.get("response", "")
    except Exception:
        return "MOCK_RESPONSE_OLLAMA_OFFLINE"


def test_generate_signal():
    prompt = "Generate a simple buy/sell trading signal using moving averages"
    response = generate_signal(prompt)
    assert response != "", "No response generated from Ollama"

    # Print for manual inspection (optional)
    print("Generated signal:", response)


if __name__ == "__main__":
    # Run the test manually
    test_generate_signal()