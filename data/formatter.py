CHAT_FORMAT_MAPPER = {}


class ChatElement(object):
    def __init__(self, content):
        self.content = content


class Token(ChatElement):
    
    def to_ids(self, tokenizer, **format_kws):
        ids = [tokenizer.convert_tokens_to_ids(self.content)]
        
        if None in ids:
            raise Exception(f"invalid Element: {self.content} ({self.type_})")
        return ids
    
    
class String(ChatElement):
    
    def to_ids(self, tokenizer, **format_kws):
        tokenizer_args = {"add_special_tokens": False}
        content = str(self.content)
        
        for k, v in format_kws.items():
            quoted_kw = "{{" + k + "}}"
            content = content.replace(quoted_kw, str(v))
            
        return tokenizer.encode(content, **tokenizer_args)
    

class Formatter(object):
    def __init__(self, name, system, prompt, sep):
        """
        sep for concat multiple-round data
        """
        self.name = name
        self.system = system
        self.prompt = prompt
        self.sep = sep
        
    def _check_session(self, sessions):
        for idx, sess in enumerate(sessions):
            assert len(sess) == 2, (
                f"Invalid round {idx}: {sess}, each sess should contains "
                "2 elements (query & resp)"
            )
            
    def truncate(self, query_ids, resp_ids, max_seq_len):
        query_len = len(query_ids)
        resp_len = len(resp_ids)
        
        if query_len + resp_len <= max_seq_len:
            return query_ids, resp_ids
        
        overflow_len = query_len + resp_len - max_seq_len
        query_cutoff = overflow_len // 2
        resp_cutoff = overflow_len - query_cutoff
        query_ids = query_ids[: -query_cutoff]
        resp_ids = resp_ids[: -resp_cutoff]
        return query_ids, resp_ids
    
    def format_to_ids(self, tokenizer, max_seq_len, **fmt_kws):
        """multi-turn formatter"""
        history = fmt_kws.get("history", [])
        sessions = history + [[fmt_kws["query"], fmt_kws["response"]]]
        self._check_session(sessions)
        
        inputs = []
        data_pairs = []
        response_ids = []
        
        for element in self.system:
            inputs += element.to_ids(tokenizer=tokenizer, **fmt_kws)
            
        for round_idx, (query, response) in enumerate(sessions):
            if round_idx > 0:
                if self.sep:
                    inputs += String(self.sep).to_ids(tokenizer=tokenizer)
                    
            for element in self.prompt:
                inputs += element.to_ids(
                    tokenizer=tokenizer, idx=round_idx, query=query,
                )
                
            if response:  # on inference mode
                curr_output = (
                    String(response).to_ids(tokenizer=tokenizer) + 
                    [tokenizer.eos_token_id]
                )
                
                inputs += curr_output
                curr_output = (
                    [-100] * (len(inputs) - len(curr_output)) + curr_output
                )
                
                response_ids = [i for i in curr_output]
                
            query_ids = [i for i in inputs]
            
            query_ids, response_ids = self.truncate(
                query_ids, response_ids, max_seq_len,
            )
            data_pairs.append([query_ids, response_ids])
            
        return data_pairs
    
    def format_to_str(self, tokenizer, max_seq_len, **fmt_kws):
        """for debugging"""
        data_pairs = self.format_to_ids(tokenizer, max_seq_len, **fmt_kws)
        decoding_args = {"skip_special_tokens": True}
        string_pairs = []
        
        for pair in data_pairs:
            pair[1] = [i for i in pair[1] if i >= 0]
            string_pairs.append(
                [tokenizer.decode(p, **decoding_args) for p in pair]
            )
        return string_pairs


def register_format(name, system, prompt, sep=None):
    CHAT_FORMAT_MAPPER[name] = Formatter(name, system, prompt, sep)


# bare formatter
register_format(
    name="base",
    system=[],
    prompt=[String("{{query}}")],
    sep="",
)

register_format(
    name="llama2",
    system=[
        String(
            "<<SYS>>\n"
            "You are a helpful, respectful and honest assistant. "
            "Always answer as helpfully as possible, while being safe. "
            "Your answers should not include any harmful, unethical, "
            "racist, sexist, toxic, dangerous, or illegal content. "
            "Please ensure that your responses are socially unbiased and "
            "positive in nature.\n\nIf a question does not make any sense, "
            "or is not factually coherent, explain why instead of answering "
            "something not correct. If you don't know the answer to a "
            "question, please don't share false information."
            "\n<</SYS>>\n\n"
        )
    ],
    prompt=[String("[INST] {{query}} [/INST]")],
    sep=None,
)

register_format(
    name="chatglm2",
    system=[Token("[gMASK]"), Token("sop")],
    prompt=[String("［Round {{idx}}］\n\n问：{{query}}\n\n答：")],
    sep="\n\n",
)

# https://github.com/THUDM/ChatGLM3/blob/main/PROMPT.md
register_format(
    name="chatglm3",
    system=[
        Token("[gMASK]"), Token("sop"), Token("<|system|>"),
        String(
            "\nYou are ChatGLM3, a large language model trained by Zhipu.AI."
            "Follow the user's systems carefully. Respond using markdown."
        ),
    ],
    prompt=[
        Token("<|user|>"), String("\n{{query}}"), Token("<|assistant|>"), 
        String("\n"),
    ]
)
    
# https://github.com/QwenLM/Qwen/blob/main/finetune.py#L125
register_format(
    name="qwen",
    system=[
        Token("<|im_start|>"), String("system\nYou are a helpful assistant."),
        Token("<|im_end|>"), String("\n"),
    ],
    prompt=[
        Token("<|im_start|>"), String("user\n{{query}}"), Token("<|im_end|>"),
        String("\n"), Token("<|im_start|>"), String("assistant\n"),
    ],
)

# https://github.com/01-ai/Yi?tab=readme-ov-file#31-use-the-chat-model
register_format(
    name="yi",
    system=[],
    prompt=[
        Token("<|im_start|>"), String("user\n{{query}}"), Token("<|im_end|>"),
        String("\n"), Token("<|im_start|>"), String("assistant\n"),
    ],
)

# TODO: not test yet
register_format(
    name="baichuan",
    system=[],
    prompt=[
        Token("<reserved_102>"), String("{{query}}"), Token("<reserved_103>"),
    ],
)

# https://github.com/baichuan-inc/Baichuan2/blob/main/fine-tune/fine-tune.py
register_format(
    name="baichuan2",
    system=[],
    prompt=[
        Token("<reserved_106>"), String("{{query}}"), Token("<reserved_107>"),
    ],
)
