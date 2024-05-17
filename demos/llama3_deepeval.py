from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base_model import DeepEvalBaseLLM
import asyncio

class Llama3_8B(DeepEvalBaseLLM):
    def __init__(
        self,
        model,
        tokenizer
    ):
        self.model = model
        self.tokenizer = tokenizer

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:

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


    async def a_generate(self, prompt: str) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.generate, prompt)

    def get_model_name(self):
        return "Llama 3 8B"

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("solidrust/Meta-Llama-3-8B-Instruct-hf-AWQ", device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained("solidrust/Meta-Llama-3-8B-Instruct-hf-AWQ", device_map="auto")

    llama_3 = Llama3_8B(model=model, tokenizer=tokenizer)
    print(llama_3.generate("Write me a joke"))