# Llama-Triton API 文件

## 1. LlamaGenerator

`class LlamaGenerator(model_dir: str|Path, device: str = 'cuda')`

用於生成文本的主要類。

### 方法

#### `generate(prompt: str, max_new_tokens: int = 32, temperature: float = 0.8, top_p: float = 0.9) -> str`

生成文本。

- **參數**:
  - `prompt`: 提示文本。
  - `max_new_tokens`: 生成的最大 token 數量。
  - `temperature`: 控制隨機性的溫度值。
  - `top_p`: 使用的 top-p 採樣閾值。
- **返回值**: 生成的文本。

## 2. LlamaConfig

`class LlamaConfig`

存放模型參數設定。

### 屬性

- `n_layers`: 模型層數。
- `n_heads`: 注意力機制的頭數。
- `n_kv_heads`: KV 注意力機制的頭數。
- `hidden_size`: 隱藏層維度。
- `intermediate_size`: 中間層維度。
- `vocab_size`: 詞彙表大小。
- `max_position_embeddings`: 最大位置編碼數量。
- `rope_freq_base`: Rotary 位置編碼基準頻率。
- `rope_freq_scale`: Rotary 頻率尺度。
- `page_size`: 每頁大小（用於 Page Attention）。
- `max_pages`: 最大頁數。
- `kv_dtype`: KV Cache 數據類型。
- `bos_token_id`: 句子起始標記的 ID。
- `eos_token_id`: 句子結束標記的 ID。
- `pad_token_id`: 填充標記的 ID。
- `tie_weights`: 是否綁定 embedding 和 LM head 的權重。
- `devices`: 使用的設備列表。

### 方法

#### `from_json(path: str|Path) -> LlamaConfig`
從 JSON 檔案讀取配置。

#### `to_json(path: str|Path)`
將配置寫入 JSON 檔案。

## 3. LlamaTritonModel

`class LlamaTritonModel(cfg: LlamaConfig, device: str = 'cuda')`

基於 Triton 加速的 Llama 模型。

### 方法

#### `forward(input_ids: torch.Tensor) -> torch.Tensor`

- **參數**:
  - `input_ids`: 輸入 token IDs。
- **返回值**: 輸出的 logits。

## 4. PageKVCache

`class PageKVCache(cfg: LlamaConfig, device: str = 'cuda')`

用於管理 Key-Value cache 的分頁記憶體。

### 方法

#### `allocate(layer: int, k: torch.Tensor, v: torch.Tensor)`

將給定的 key 和 value 存入指定層的 KV cache。

- **參數**:
  - `layer`: 指定的層索引。
  - `k`: Key 張量。
  - `v`: Value 張量。

#### `get_kv_pages(layer: int, device: Optional[str]) -> Tuple[torch.Tensor, torch.Tensor]`

獲取指定層的 KV cache 頁面。

- **參數**:
  - `layer`: 指定的層索引。
  - `device`: 指定返回張量的設備。
- **返回值**: Key 和 Value 張量。

## 5. SentencePieceTokenizer

`class SentencePieceTokenizer(model_path: str)`

SentencePiece 的 tokenizer 實現。

### 方法

#### `encode(text: str, add_bos: bool = True, add_eos: bool = False) -> List[int]`

將文本編碼為 token IDs。

- **參數**:
  - `text`: 輸入文本。
  - `add_bos`: 是否添加 BOS token。
  - `add_eos`: 是否添加 EOS token。
- **返回值**: Token ID 列表。

#### `decode(ids: List[int]) -> str`

將 token IDs 解碼為文本。

- **參數**:
  - `ids`: Token ID 列表。
- **返回值**: 解碼後的文本。

## 6. Triton Kernels

### matmul

`triton_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor`

基於 Triton 的 GEMM 運算。

- **參數**:
  - `a`: 左邊矩陣。
  - `b`: 右邊矩陣。
- **返回值**: GEMM 結果。

### page_attention

`page_attention(q: torch.Tensor, kv_pages: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor`

實現 Triton 版的 Page Attention。

- **參數**:
  - `q`: Query 張量。
  - `kv_pages`: Key 和 Value 張量。
- **返回值**: Attention 結果。

### rmsnorm

`rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor`

基於 Triton 的 RMS Normalization。

- **參數**:
  - `x`: 輸入張量。
  - `w`: 權重張量。
  - `eps`: 防止除零的小常數。
- **返回值**: Normalization 結果。

### rotary

#### `build_rotary_cache(base: float, dim: int, max_pos: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]`

建立 rotary positional embedding 快取。

#### `apply_rotary(t: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, pos: int) -> torch.Tensor`

應用 rotary positional embedding。

- **參數**:
  - `t`: 輸入張量。
  - `sin`: Sin cache 張量。
  - `cos`: Cos cache 張量。
  - `pos`: 位置索引。
- **返回值**: 旋轉後的張量。

