from fastapi import FastAPI, Request
from pydantic import BaseModel
from vllm import LLM, SamplingParams
import os

app = FastAPI()

gpu_id = os.getenv("GPU_ID", "0")
model = LLM(model="google/gemma-3-27b-it", gpu_memory_utilization=0.9, tensor_parallel_size=2)

class PromptMessage(BaseModel):
    role: str
    content: str

class GenerateRequest(BaseModel):
    messages: list[PromptMessage]
    max_tokens: int = 10
    temperature: float = 0.5

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
