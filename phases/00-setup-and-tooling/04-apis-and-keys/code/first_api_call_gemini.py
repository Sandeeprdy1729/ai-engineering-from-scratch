import os       # Access environment variables (like GEMINI_API_KEY)
import json      # Convert Python dicts to JSON strings and parse JSON responses
import urllib.request  # Built-in HTTP client — no external dependencies needed for raw requests


def call_with_sdk():
    """Make an API call using Google's official Python SDK (google-genai).
    
    The SDK handles: authentication, HTTP connection, request formatting,
    response parsing, error handling, and retries — all automatically.
    """
    try:
        from google import genai   # Third-party SDK from Google
    except ImportError:
        print("Install the SDK: pip install google-genai")
        return

    # Create a client — reads the API key from the argument (or env var GOOGLE_API_KEY by default)
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    # .generate_content() is a high-level method — it builds the HTTP request,
    # sends it, parses the response, and returns a Python object
    response = client.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents="What is a neural network in one sentence?",
    )
    # response.text gives the model's reply directly — no JSON parsing needed
    print(f"SDK response: {response.text}")
    # usage_metadata has token counts — the SDK extracts them from the raw response
    usage = response.usage_metadata
    if usage:
        print(f"Tokens used: {usage.prompt_token_count} in, {usage.candidates_token_count} out")
    # The SDK also handles: authentication headers, retries on 429/503, streaming, etc.


def call_raw_http():
    """Make the same API call using raw HTTP — no SDK, no third-party imports.
    
    This is what the SDK does internally. Seeing it demystifies the "magic".
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Set GEMINI_API_KEY environment variable first")
        return

    model = "gemini-2.0-flash-lite"
    # The full REST endpoint URL. Every SDK call maps to one of these.
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    # HTTP headers — we must set Content-Type ourselves
    headers = {"Content-Type": "application/json"}
    # Build the request body manually as a JSON string, then encode to bytes
    body = json.dumps({
        "contents": [{"parts": [{"text": "What is a neural network in one sentence?"}]}]
    }).encode()

    # Create a POST request object with URL, body, headers, and method
    req = urllib.request.Request(
        url, data=body, headers=headers, method="POST"
    )
    # Send the request and read the response
    with urllib.request.urlopen(req) as resp:
        # Parse the raw JSON response into a Python dict
        result = json.loads(resp.read())
        # Navigate nested dicts to extract the generated text
        text = result["candidates"][0]["content"]["parts"][0]["text"]
        print(f"Raw HTTP response: {text}")
    # Notice: we didn't handle auth headers, retries, or error formatting here.
    # The SDK does all of that for you.


if __name__ == "__main__":
    print("=== Gemini API Calls ===\n")
    print("1. Using the SDK:")
    call_with_sdk()
    print("\n2. Using raw HTTP:")
    call_raw_http()
