import logging


logger = logging.getLogger(__name__)

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
        """Truncate query_ids and resp_ids pair given max_seq_len

        :returns
            truncated (bool): mark if inputs was truncated
            query_ids (List[int]): trunceted query_ids
            resp_ids (List[int]): truncated resp_ids
        """
        query_len = len(query_ids)

        if query_len <= max_seq_len:
            return False, query_ids, resp_ids
        
        resp_len = len(resp_ids)
        if resp_len > 0:
            assert query_len == resp_len, (
                f"query ({query_len}) and resp ({resp_len}) length mismatch!"
            )

            # TODO: 大概率会丢掉template 部分内容，此种截断方式不合理
            query_ids = query_ids[-max_seq_len:]
            resp_ids = resp_ids[-max_seq_len:]
        else:
            query_ids = query_ids[-max_seq_len:]

        return True, query_ids, resp_ids
    
    def format_to_ids(self, tokenizer, max_seq_len, **fmt_kws):
        """multi-turn formatter"""
        history = fmt_kws.get("history", [])

        sessions = history + [[fmt_kws["query"], fmt_kws["response"]]]
        self._check_session(sessions)
        
        inputs = []
        data_pairs = []
        response_ids = []

        strings = []
        
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
            
            is_truncated, query_ids, response_ids = self.truncate(
                query_ids, response_ids, max_seq_len,
            )

            data_pairs.append([query_ids, response_ids])

            if is_truncated:
                break

        return data_pairs
    
    def format_to_str(self, tokenizer, max_seq_len, **fmt_kws):
        """for debugging"""

        session_ids = self.format_to_ids(tokenizer, max_seq_len, **fmt_kws)

        session_strings = [
            tokenizer.batch_decode(sess) for sess in session_ids
        ]

        return session_strings


def register_format(name, system=None, prompt=None, sep=None, _copy_from=None):
    if not system:
        system = []

    if not prompt:
        prompt = []
        
    if _copy_from:
        assert _copy_from in CHAT_FORMAT_MAPPER, f"{_copy_from} not exists!"
        CHAT_FORMAT_MAPPER[name] = CHAT_FORMAT_MAPPER[_copy_from]
    else:
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

# qwen2: https://huggingface.co/Qwen/Qwen2-7B-Instruct/blob/main/tokenizer_config.json
register_format(
    name="qwen2",
    _copy_from="qwen",
)
