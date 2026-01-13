import os
import json
import torch
import torch.nn as nn
from transformers import (
    PreTrainedModel,
    GenerationMixin,
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM,
)
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union


# ================== 虚拟字节级 Tokenizer ==================

class ByteTokenizer:
    """
    虚拟字节级 Tokenizer，只做简单的 UTF-8 编码/解码，
    目的是让无分词模型也能挂到 HF 的 pipeline 上。
    """

    def __init__(self):
        self.vocab_size = 256  # 0..255 字节
        self.eos_token = "\x00"
        self.pad_token = "\x00"
        self.model_max_length = 2048
        self.is_fast = True
        self.name_or_path = "byte-tokenizer"

    def encode(self, text: str) -> List[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: List[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")

    def __call__(
        self,
        texts: Union[str, List[str]],
        *args,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        模拟 HF Tokenizer 的返回格式：
        {
          "input_ids": LongTensor[B, L],
          "attention_mask": LongTensor[B, L]
        }
        """
        if isinstance(texts, str):
            texts = [texts]

        encoded = [list(t.encode("utf-8")) for t in texts]
        max_len = max(len(seq) for seq in encoded)
        padded = [seq + [0] * (max_len - len(seq)) for seq in encoded]

        input_ids = torch.tensor(padded, dtype=torch.long)
        attn_mask = torch.tensor([[1] * len(seq) for seq in encoded], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attn_mask}

    @property
    def vocab(self) -> Dict[str, int]:
        return {chr(i): i for i in range(256)}

    @property
    def inv_vocab(self) -> Dict[int, str]:
        return {i: chr(i) for i in range(256)}


# ================== 配置类 ==================

@dataclass
class CharEngramConfig(PretrainedConfig):
    """
    字节级 Engram 模型配置，兼容 HF PretrainedConfig。
    """

    model_type: str = "char-engram"

    # 模型超参
    ngram_size: int = 3
    memory_dim: int = 128
    vocab_size: int = 256
    hidden_size: int = 256
    num_layers: int = 4
    memory_capacity: int = 100_000
    dropout: float = 0.1
    max_seq_len: int = 128
    engram_layer_index: int = 1

    # HF 期望的一些属性
    _attn_implementation: str = "eager"
    is_encoder_decoder: bool = False

    def __init__(self, **kwargs):
        # 让 from_dict / from_pretrained 能安全传入 model_type / 其他字段
        # 这里简单地：只接收本类声明过的字段，忽略其余字段
        field_names = {
            "model_type",
            "ngram_size",
            "memory_dim",
            "vocab_size",
            "hidden_size",
            "num_layers",
            "memory_capacity",
            "dropout",
            "max_seq_len",
            "engram_layer_index",
            "_attn_implementation",
            "is_encoder_decoder",
        }
        for k, v in kwargs.items():
            if k in field_names:
                setattr(self, k, v)

        # 让父类记录 config_dict（会用于 save_pretrained）
        super().__init__(**kwargs)

        # HF 内部会访问 self._attn_implementation_internal
        self._attn_implementation_internal = self._attn_implementation

        # 一些简单的 sanity check
        if not (2 <= self.ngram_size <= 10):
            raise ValueError("ngram_size 必须在 [2, 10] 范围内")
        if self.hidden_size % 64 != 0:
            raise ValueError("hidden_size 建议为 64 的倍数")


# ================== 字节级嵌入 ==================

class ByteEmbedding(nn.Module):
    """字节级嵌入 + 位置编码"""

    def __init__(self, config: CharEngramConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position = nn.Embedding(config.max_seq_len, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_bytes: torch.Tensor) -> torch.Tensor:
        # input_bytes: [B, L]
        B, L = input_bytes.shape
        pos_ids = torch.arange(L, device=input_bytes.device).unsqueeze(0).expand(B, -1)
        return self.dropout(self.embedding(input_bytes) + self.position(pos_ids))


# ================== N-gram 哈希映射 ==================

class HashMapping(nn.Module):
    """把字节 N-gram 映射到 [0, capacity) 的下标"""

    def __init__(self, capacity: int):
        super().__init__()
        self.capacity = capacity
        self.register_buffer("seed", torch.randint(0, 2**31, (1,)))

    def hash_ngram(self, ngram_bytes: bytes) -> int:
        return hash(ngram_bytes + self.seed.item().to_bytes(4, "big")) % self.capacity

    def batch_hash(self, input_bytes: torch.Tensor, ngram_size: int) -> torch.Tensor:
        B, L = input_bytes.shape
        out_len = max(0, L - ngram_size + 1)
        if out_len == 0:
            return torch.empty(0, dtype=torch.long, device=input_bytes.device)

        hashes = torch.zeros(B, out_len, dtype=torch.long, device=input_bytes.device)
        for b in range(B):
            row = input_bytes[b].cpu().numpy()
            for pos in range(out_len):
                ngram_bytes = bytes(row[pos : pos + ngram_size])
                hashes[b, pos] = self.hash_ngram(ngram_bytes)
        return hashes


# ================== 字节 N-gram 记忆表 ==================

class ByteMemory(nn.Module):
    """静态字节 N-gram 记忆表"""

    def __init__(self, config: CharEngramConfig):
        super().__init__()
        self.config = config
        self.hash_map = HashMapping(config.memory_capacity)
        self.memory_table = nn.Parameter(
            torch.empty(config.memory_capacity, config.memory_dim)
        )
        nn.init.kaiming_uniform_(self.memory_table)

    def forward(self, input_bytes: torch.Tensor) -> torch.Tensor:
        N = self.config.ngram_size
        B, L = input_bytes.shape
        out_len = max(0, L - N + 1)
        if out_len == 0:
            return torch.empty(0, 0, self.config.memory_dim, device=input_bytes.device)

        hashes = self.hash_map.batch_hash(input_bytes, N)  # [B, L-N+1]
        return self.memory_table[hashes]  # [B, L-N+1, memory_dim]


# ================== 门控融合 ==================

class GateFusion(nn.Module):
    """hidden 与 memory 的门控融合"""

    def __init__(self, hidden_size: int, memory_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size + memory_dim, hidden_size)
        self.act = nn.Sigmoid()

    def forward(self, hidden: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # hidden: [B, L, H], memory: [B, Lm, M]
        B, L, H = hidden.shape
        Bm, Lm, M = memory.shape
        L_eff = min(L, Lm)

        h = hidden[:, :L_eff, :]
        m = memory[:, :L_eff, :]

        x = torch.cat([h, m], dim=-1)
        gate = self.act(self.proj(x))
        out = gate * h + (1.0 - gate) * m

        hidden = hidden.clone()
        hidden[:, :L_eff, :] = out
        return hidden


# ================== Transformer Block ==================

class CharTransformerBlock(nn.Module):
    """简化版 Transformer encoder block"""

    def __init__(self, config: CharEngramConfig):
        super().__init__()
        num_heads = max(1, config.hidden_size // 64)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=num_heads,
            batch_first=True,
            dropout=config.dropout,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
        )
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)
        return x


# ================== Engram 模块封装 ==================

class CharEngram(nn.Module):
    """在某一层插入的 Engram 模块"""

    def __init__(self, config: CharEngramConfig):
        super().__init__()
        self.config = config
        self.memory = ByteMemory(config)
        self.value_proj = nn.Linear(config.memory_dim, config.hidden_size)
        self.fusion = GateFusion(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor, input_bytes: torch.Tensor) -> torch.Tensor:
        mem = self.memory(input_bytes)
        if mem.numel() == 0:
            return hidden_states
        mem_proj = self.value_proj(mem)
        return self.fusion(hidden_states, mem_proj)


# ================== 输出类型 ==================

class CausalLMOutputWithHiddenStates(dict):
    """最小化的 CausalLM 输出，满足 GenerationMixin 期望"""

    def __init__(self, logits=None, hidden_states=None):
        super().__init__(logits=logits, hidden_states=hidden_states)
        self.logits = logits
        self.hidden_states = hidden_states


# ================== 主模型类 ==================

class CharLevelEngramModel(PreTrainedModel, GenerationMixin):
    """
    字节级 Engram 模型（HF 兼容版本）

    - 继承 PreTrainedModel + GenerationMixin，可直接使用 model.generate。
    - 输入为字节序列，不依赖分词器。
    """

    config_class = CharEngramConfig

    def __init__(self, config: CharEngramConfig):
        super().__init__(config)
        self.config = config

        self.embed = ByteEmbedding(config)
        self.layers = nn.ModuleList(
            [CharTransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.engram = CharEngram(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        self.post_init()

    def post_init(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.hidden_size ** -0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.hidden_size ** -0.5)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithHiddenStates:
        if input_ids is None:
            raise ValueError("必须提供 input_ids（字节序列）")

        if input_ids.max() >= self.config.vocab_size or input_ids.min() < 0:
            raise ValueError(
                f"input_ids 必须在 [0, {self.config.vocab_size - 1}] 范围内（字节级）"
            )

        hidden = self.embed(input_ids)

        for i, layer in enumerate(self.layers):
            hidden = layer(hidden)
            if i == self.config.engram_layer_index:
                hidden = self.engram(hidden, input_ids)

        logits = self.lm_head(hidden)
        return CausalLMOutputWithHiddenStates(logits=logits, hidden_states=hidden)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        **kwargs,
    ) -> Dict[str, Any]:
        # 简单场景下，只需要把 input_ids 传回去即可
        return {"input_ids": input_ids}

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        # 通过 HF 的 Config 加载
        config = CharEngramConfig.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)

        weights_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        if os.path.exists(weights_path):
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        return model

    def save_pretrained(self, save_directory: str, *args, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        with open(os.path.join(save_directory, "model.config"), "w") as f:
            json.dump({"model_type": "char-engram"}, f)
        print(f"[INFO] 模型已保存到 {save_directory}")


# ================== 注册到 Auto* ==================

AutoConfig.register("char-engram", CharEngramConfig)
AutoModelForCausalLM.register(CharEngramConfig, CharLevelEngramModel)


# ================== 自检 Demo（可选） ==================

if __name__ == "__main__":
    print("=" * 80)
    print("HF 兼容的字节级 Engram 模型演示")
    print("=" * 80)

    # 1. 创建配置和模型
    print("\n1. 创建配置和模型")
    config = CharEngramConfig(
        ngram_size=4,
        hidden_size=128,
        num_layers=3,
        vocab_size=256,
        max_seq_len=128,
    )
    model = CharLevelEngramModel(config)
    print(f"  模型参数总数: {sum(p.numel() for p in model.parameters()):,}")

    # 2. 测试前向
    print("\n2. 测试前向传播")
    texts = [
        "Hello, World! Byte-level Engram demo.",
        "无分词架构：我们直接在 UTF-8 字节上做 N-gram 记忆",
        "DeepSeek Engram: Conditional Memory via Scalable Lookup",
    ]
    input_ids = []
    for t in texts:
        b = list(t.encode("utf-8"))
        input_ids.append(b + [0] * (config.max_seq_len - len(b)))
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    print(f"  输入 shape: {input_ids.shape}")

    with torch.no_grad():
        outputs = model(input_ids)
    print(f"  输出 logits shape: {outputs.logits.shape}")

    # 3. 保存 / 加载
    print("\n3. 测试保存/加载")
    model.save_pretrained("./char_engram_model")
    loaded_model = CharLevelEngramModel.from_pretrained("./char_engram_model")
    print("  模型成功保存和加载！")

    # 4. 生成
    print("\n4. 演示生成")
    tokenizer = ByteTokenizer()
    prompt = "DeepSeek Engram is"
    print("  Prompt:", prompt)
    batch = tokenizer(prompt)
    prompt_ids = batch["input_ids"]

    with torch.no_grad():
        generated = loaded_model.generate(
            prompt_ids,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            pad_token_id=0,
        )

    text_out = tokenizer.decode(generated[0].tolist())
    print("  生成结果:", text_out)
    print("=" * 80)
