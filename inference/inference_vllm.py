import sys
sys.path.append("../../")
from vllm import LLM, SamplingParams
from llmx.data.formatter import CHAT_FORMAT_MAPPER


class VllmPredictor(object):

    def __init__(
        self, model_name_or_path, tp_size, enable_prefix_caching, max_tokens,
        template_name,
    ):
        self.llm = LLM(
            model=model_name_or_path,
            trust_remote_code=True, 
            tensor_parallel_size=tp_size,
            enable_prefix_caching=enable_prefix_caching,
            gpu_memory_utilization=0.98,
            use_v2_block_manager=False,
            enforce_eager=True,
        )

        self.sampling_params = SamplingParams(
            temperature=0.9, top_p=0.9, max_tokens=max_tokens,
            
        )

        self.tokenizer = self.llm.get_tokenizer()

        self.formatter = CHAT_FORMAT_MAPPER.get(template_name, None)
    
    def get_outputs(self, vllm_outputs):
        final_outputs = []

        for output in vllm_outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            final_outputs.append(generated_text)
        
        return final_outputs
    
    def apply_template(self, queries, max_seq_len):
        prompts = []

        for query in queries:
            if self.formatter is not None:
                prompts.append(
                    self.formatter.format_to_str(
                        tokenizer=self.tokenizer, max_seq_len=max_seq_len,
                        history=[], query=query, response="",
                    )[0][0]
                )

            else:
                conversation = [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant"
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ]
                
                prompts.append(
                    self.tokenizer.apply_chat_template(
                        conversation=conversation,
                        tokenize=False,
                    )
                )
        return prompts
    
    def generate(self, queries, max_seq_len):
        prompts = self.apply_template(queries, max_seq_len)

        vllm_outputs = self.llm.generate(
            prompts, self.sampling_params,
        )

        for idx, out in enumerate(vllm_outputs):
            print(f"{idx} th output:\ntoken_ids len: {len(out.prompt_token_ids)}\nmetrics: {out.metrics}")
            generate_text = out.outputs[0].text

            arrival_time = out.metrics.arrival_time
            first_token_time = out.metrics.first_token_time
            finished_time = out.metrics.finished_time

            ttft = first_token_time - arrival_time
            tpt = (finished_time - first_token_time) / (len(generate_text) + 1)
            print(f"ttft: {ttft}, tpt: {tpt}")
            print(generate_text)
            print("----")

        outputs = self.get_outputs(vllm_outputs)

        return outputs
