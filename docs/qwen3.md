# qwen3

> Reference: <https://arxiv.org/pdf/2505.02214v1>

## Table4: 1 to 8-bits per-group PTQ results of Qwen3 Models

> select "Method: AWQ"

| Model | #W | Wiki2(↓) | c4(↓) | PiQA | ArcE | ArcC | HellaS | WinoG | BoolQ | Avg(↑) | MMLU |
|-------|----|----------|-------|------|------|------|--------|-------|-------|--------|------|
| 4B    | 8  | 13.6     | 16.6  | 74.9 | 80.6 | 50.5 | 52.3   | 65.6  | 85.1  | 68.1   | 69.6 |
| 4B    | 4  | 18.1     | 19.6  | 72.7 | 73.0 | 45.0 | 48.6   | 60.1  | 82.0  | 63.6   | 64.1 |
| 8B    | 8  | 9.72     | 13.3  | 76.8 | 83.5 | 55.5 | 57.1   | 67.5  | 86.5  | 71.2   | 74.5 |
| 8B    | 4  | 11.3     | 14.8  | 75.5 | 80.8 | 48.8 | 54.6   | 64.7  | 82.1  | 67.8   | 69.3 |
| 14B   | 8  | 8.63     | 12.0  | 80.0 | 84.3 | 58.4 | 60.9   | 72.8  | 89.5  | 74.3   | 78.5 |
| 14B   | 4  | 9.48     | 13.3  | 77.3 | 79.5 | 51.9 | 58.4   | 70.0  | 87.0  | 70.7   | 75.9 |

Notice, GGUF model size:

* 4b-q8_0(4.4GB) vs 8b-q4_K_M(5.2GB)
* 8b-q8_0(8.9GB) vs 14b-q4_K_M(9.3GB)

## Local speed test (same runner)

Q: `extract keywords, rather than answering question, from "qwen3在同参数量时的Q4和Q8之间性能差距多少？" /think`

* qwen3:4b-q8_0    -> 32.15 tokens/s
* qwen3:8b-q4_K_M  -> 23.18 tokens/s
* qwen3:14b-q4_K_M -> 13.85 tokens/s
