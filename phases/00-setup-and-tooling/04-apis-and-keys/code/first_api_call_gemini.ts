// Phase 0 · Lesson 04 — APIs and keys (Gemini TypeScript port).
// Reads GEMINI_API_KEY from env, then makes one generateContent call using
// both the official SDK and raw HTTP. Set MOCK=1 to skip the network.
// Refs: https://ai.google.dev/gemini-api/docs/quickstart
//       https://nodejs.org/api/globals.html#fetch (Node 18+ ships fetch)

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import process from "node:process";

type GenerateContentResponse = {
  candidates: { content: { parts: { text: string }[] } }[];
  usageMetadata?: { promptTokenCount: number; candidatesTokenCount: number };
};

const MOCK_RESPONSE: GenerateContentResponse = {
  candidates: [
    {
      content: {
        parts: [
          {
            text: "A neural network is a stack of differentiable functions that learns patterns by adjusting weights against a loss signal.",
          },
        ],
      },
    },
  ],
  usageMetadata: { promptTokenCount: 12, candidatesTokenCount: 28 },
};

// .env loader — same pattern as first_api_call.ts. KEY=VALUE per line, # comments.
function loadDotenv(path: string): Record<string, string> {
  let raw: string;
  try {
    raw = readFileSync(path, "utf8");
  } catch {
    return {};
  }
  const out: Record<string, string> = {};
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) continue;
    const eq = trimmed.indexOf("=");
    if (eq <= 0) continue;
    const key = trimmed.slice(0, eq).trim();
    let value = trimmed.slice(eq + 1).trim();
    if (
      (value.startsWith('"') && value.endsWith('"')) ||
      (value.startsWith("'") && value.endsWith("'"))
    ) {
      value = value.slice(1, -1);
    }
    out[key] = value;
  }
  return out;
}

function mergeEnv(): Record<string, string | undefined> {
  const fromFile = loadDotenv(resolve(process.cwd(), ".env"));
  return { ...fromFile, ...process.env };
}

function buildGeminiUrl(model: string): string {
  // The full REST endpoint. SDKs build this URL internally.
  return `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`;
}

function buildGeminiBody(prompt: string): string {
  // The JSON payload — manually constructed here, built by the SDK in callWithSdk.
  return JSON.stringify({
    contents: [{ parts: [{ text: prompt }] }],
  });
}

/** Make an API call using the official Google Gen AI SDK (@google/genai).
 *
 * The SDK handles: authentication, HTTP connection, request formatting,
 * response parsing, error handling, and retries — all automatically. */
async function callWithSdk(
  apiKey: string,
  prompt: string,
  model: string,
): Promise<number> {
  if (process.env.MOCK === "1" || apiKey === "mock") {
    const text = MOCK_RESPONSE.candidates[0].content.parts[0].text;
    process.stdout.write(`SDK response: ${text}\n`);
    return 0;
  }
  try {
    // Dynamic import so the script doesn't crash if the SDK isn't installed.
    const { GoogleGenAI } = await import("@google/genai");
    const client = new GoogleGenAI({ apiKey });
    const response = await client.models.generateContent({
      model,
      contents: prompt,
    });
    // response.text gives the model's reply directly — no JSON parsing needed.
    const text = response.text ?? "";
    process.stdout.write(`SDK response: ${text}\n`);
    if (response.usageMetadata) {
      process.stdout.write(
        `Tokens used: ${response.usageMetadata.promptTokenCount} in, ` +
          `${response.usageMetadata.candidatesTokenCount} out\n`,
      );
    }
    return 0;
  } catch (err) {
    process.stderr.write(`SDK call failed: ${(err as Error).message}\n`);
    return 1;
  }
}

/** Make the same API call using raw HTTP fetch — no SDK, no third-party imports.
 *
 * This is what the SDK does internally. Seeing it demystifies the "magic". */
async function callRawHttp(
  apiKey: string,
  prompt: string,
  model: string,
): Promise<number> {
  if (process.env.MOCK === "1" || apiKey === "mock") {
    // Use the fixture so MOCK=1 returns the same shape as a real response.
    const text = MOCK_RESPONSE.candidates[0].content.parts[0].text;
    process.stdout.write(`Raw HTTP response: ${text}\n`);
    return 0;
  }

  try {
    const url = buildGeminiUrl(model);
    // HTTP headers — we must set Content-Type ourselves; API key goes as a query param.
    const headers = { "Content-Type": "application/json" };
    // Build and encode the request body manually
    const body = buildGeminiBody(prompt);

    // Fetch is built into Node 18+. This is exactly what the SDK wraps.
    const resp = await fetch(`${url}?key=${apiKey}`, {
      method: "POST",
      headers,
      body,
    });

    if (!resp.ok) {
      const bodyText = await resp.text();
      throw new Error(`Gemini ${resp.status}: ${bodyText.slice(0, 200)}`);
    }
    // Parse the raw JSON response into a typed object
    const result = (await resp.json()) as GenerateContentResponse;
    // Navigate nested objects to extract the generated text
    const text = result.candidates[0].content.parts[0].text;
    process.stdout.write(`Raw HTTP response: ${text}\n`);
    if (result.usageMetadata) {
      process.stdout.write(
        `Tokens used: ${result.usageMetadata.promptTokenCount} in, ` +
          `${result.usageMetadata.candidatesTokenCount} out\n`,
      );
    }
    return 0;
  } catch (err) {
    process.stderr.write(`Raw HTTP call failed: ${(err as Error).message}\n`);
    return 1;
  }
}

async function main(): Promise<number> {
  const env = mergeEnv();
  const apiKey = env.GEMINI_API_KEY ?? "mock";
  const usingMock = process.env.MOCK === "1" || apiKey === "mock";
  const model = "gemini-2.0-flash-lite";
  const prompt = "What is a neural network in one sentence?";

  process.stdout.write("=== Gemini API Calls ===\n\n");
  process.stdout.write(
    usingMock
      ? "Mode: MOCK (no network). Unset MOCK and export GEMINI_API_KEY for a live call.\n\n"
      : "Mode: LIVE.\n\n",
  );

  process.stdout.write("1. Using the SDK:\n");
  const sdkCode = await callWithSdk(apiKey, prompt, model);

  process.stdout.write("\n2. Using raw HTTP:\n");
  const httpCode = await callRawHttp(apiKey, prompt, model);

  return sdkCode === 0 && httpCode === 0 ? 0 : 1;
}

main().then((code) => process.exit(code));
