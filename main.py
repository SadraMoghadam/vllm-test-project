from fastapi import FastAPI, Request
from pydantic import BaseModel
from vllm import LLM, SamplingParams

app = FastAPI()
model = LLM(model="openai-community/gpt2", device="cpu")

class PromptMessage(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    messages: list[PromptMessage]
    max_tokens: int = 200
    temperature: float = 0.7

@app.post("/chat")
async def generate_text(request: GenerateRequest):
    prompt = ""
    for msg in request.messages:
        prompt += f"<|{msg.role}|>\n{msg.content}\n"

    sampling_params = SamplingParams(
        max_tokens=request.max_tokens,
        temperature=request.temperature
    )

    outputs = model.generate(prompt, sampling_params)
    return {"response": outputs[0].outputs[0].text}
