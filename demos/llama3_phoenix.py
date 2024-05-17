from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
import asyncio
from langchain_core.outputs import LLMResult
from phoenix.evals.models import BaseModel
from typing import Optional
from dataclasses import field, dataclass
from phoenix.evals.models.rate_limiters import RateLimiter



class LLamaRateLimitError(Exception):
    pass

@dataclass
class Llama3_8B(BaseModel):
    def __init__(
        self,
        model,
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer
        self._init_rate_limiter()

    def load_model(self):
        return self.model
    
    def _init_rate_limiter(self) -> None:
        self._rate_limiter = RateLimiter(
            rate_limit_error=LLamaRateLimitError,
            max_rate_limit_retries=10,
            initial_per_second_request_rate=1,
            maximum_per_second_request_rate=20,
            enforcement_window_minutes=1,
        )

    def _generate(self, prompt: str, instruction: Optional[str] = None) -> str:

        model = self.load_model()

        system_prompt = "You are a helpful AI assistant."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(model.device)

        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        outputs = model.generate(
            input_ids,
            max_new_tokens=2048,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        ).to(model.device)

        response = outputs[0][input_ids.shape[-1]:]

        return self.tokenizer.decode(response, skip_special_tokens=True)


    async def _async_generate(self, prompt: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.generate, prompt)

    def _model_name(self) -> str:
        return "Llama 3 8B Instruct AWQ"


if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("solidrust/Meta-Llama-3-8B-Instruct-hf-AWQ", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("solidrust/Meta-Llama-3-8B-Instruct-hf-AWQ", device_map="auto")

    llama_3 = Llama3_8B(model=model, tokenizer=tokenizer)
    print(llama_3.generate_text("Write me a joke"))