# Kimi K3.1 在线 LoRA

本目录仅保留 MindSpeed-MM adapter 转换入口。运行时实现位于 `vllm_ascend/lora/kimi/`。
测试、调试、FORGE 评测和历史产物已移出本仓库；完整记录保留在原交付目录。

## 启用

安装本仓库后设置：

```bash
export VLLM_ASCEND_ENABLE_KIMI_LORA=1
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
```

若设置了 `VLLM_PLUGINS` 白名单，加入 `ascend_kimi_lora`。
不要同时启用旧的外置 `kimi_online_lora` plugin。

服务沿用既有模型、网络及并行启动配置，增加：

```text
--enable-lora --max-lora-rank 8 --max-loras 1 --max-cpu-loras 3
--enforce-eager --dtype bfloat16 --no-enable-prefix-caching --max-num-seqs 1
```

支持范围限定为 TP8、DP1/2/16、PP/PCP/DCP=1、BF16、native/eager、单活动 adapter。
开启适配时自动使用 Kimi LoRA 的 ALLGATHER 通信和非融合 O projection 路径。
不支持量化、graph、fully-sharded LoRA、speculative decoding；不支持项明确报错。

## 转换与在线加载

```bash
python convert_adapter.py --source /path/to/lora_adapter_iteration_200.safetensors \
  --output /path/to/adapter_200
curl -f http://127.0.0.1:18083/v1/load_lora_adapter -H 'Content-Type: application/json' \
  -d '{"lora_name":"kimi-step-200","lora_path":"/path/to/adapter_200"}'
```

源文件同目录需包含 `adapter_config.json`，输出目录必须不存在。
转换保留原权重，只改变参数命名和文件布局。请求的 `model` 使用 `kimi-step-200`。
卸载使用 `/v1/unload_lora_adapter`，参数为 `{"lora_name":"kimi-step-200"}`。

## 实现

- `registration.py`：模型能力、packed/fused 投影包装、目标模块覆盖检查及 manager 接入。
- `sfa.py`：DSA 吸收后的 K/V 增量、TP 切片、adapter 激活/释放与失败回滚。
- `native_linear.py`：LoRA 线性增量；`token_slots.py`：padding 和 TP/SP token 映射。
- `idle_dp.py`：空闲 DP 使用基座映射，保留真实 adapter；`configuration.py`：支持范围校验。
- `__init__.py`：默认关闭的 plugin 入口；worker patch 安装 runner 生命周期处理。

当前仓库基于新版 MRv2，尚需重新完成 NPU 端到端验收；原镜像上的评测不替代此验收。
