import cudnn
import torch
import random
import time

def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)

RAGGED_UID = 1
Q_UID = 2
K_UID = 3
V_UID = 4
SEQLEN_Q_UID = 5
SEQLEN_KV_UID = 6
PAGE_TABLE_K_UID = 7
PAGE_TABLE_V_UID = 8
O_UID = 9


def build_graph(
        batch_size,
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        num_blocks,
        max_input_len,
        max_seq_len,
    ):
    max_block_per_seq = max_seq_len // block_size

    q_dims = (batch_size, num_heads, max_input_len, head_size)
    q_strides = (num_heads * head_size * max_input_len, head_size, head_size * num_heads, 1)
    paged_k_dims = (num_blocks, num_kv_heads, block_size, head_size)
    paged_k_strides = (num_kv_heads * block_size * head_size, block_size * head_size, head_size, 1)
    paged_v_dims = (num_blocks, num_kv_heads, block_size, head_size)
    paged_v_strides = (num_kv_heads * block_size * head_size, block_size * head_size, head_size, 1)
    seq_q_dims = (batch_size, 1, 1, 1)
    seq_q_strides = (1, 1, 1, 1)
    seq_kv_dims = (batch_size, 1, 1, 1)
    seq_kv_strides = (1, 1, 1, 1)
    page_table_dims = (batch_size, 1, max_block_per_seq, 1)
    page_table_strides = (max_block_per_seq, max_block_per_seq, 1, 1)
    ragged_offset_dims = (batch_size + 1, 1, 1, 1)
    ragged_offset_strides = (1, 1, 1, 1)


    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.BFLOAT16,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    
    ragged_offset = graph.tensor(
        name="ragged_offset",
        dim=ragged_offset_dims,
        stride=ragged_offset_strides,
        data_type=cudnn.data_type.INT32,
    )
    ragged_offset.set_uid(RAGGED_UID)

    q = graph.tensor(
        name="q",
        dim=q_dims,
        stride=q_strides,
        data_type=cudnn.data_type.BFLOAT16,
        ragged_offset=ragged_offset,
    )
    q.set_uid(Q_UID)

    k = graph.tensor(
        name="k",
        dim=paged_k_dims,
        stride=paged_k_strides,
        data_type=cudnn.data_type.BFLOAT16,
    )
    k.set_uid(K_UID)

    v = graph.tensor(
        name="v",
        dim=paged_v_dims,
        stride=paged_v_strides,
        data_type=cudnn.data_type.BFLOAT16,
    )
    v.set_uid(V_UID)

    seqlen_q = graph.tensor(
        name="seqlen_q",
        dim=seq_q_dims,
        stride=seq_q_strides,
        data_type=cudnn.data_type.INT32,
    )
    seqlen_q.set_uid(SEQLEN_Q_UID)

    seqlen_kv = graph.tensor(
        name="seqlen_kv",
        dim=seq_kv_dims,
        stride=seq_kv_strides,
        data_type=cudnn.data_type.INT32,
    )
    seqlen_kv.set_uid(SEQLEN_KV_UID)

    page_table_k = graph.tensor(
        name="page_table_k",
        dim=page_table_dims,
        stride=page_table_strides,
        data_type=cudnn.data_type.INT32,
    )
    page_table_k.set_uid(PAGE_TABLE_K_UID)

    page_table_v = graph.tensor(
        name="page_table_v",
        dim=page_table_dims,
        stride=page_table_strides,
        data_type=cudnn.data_type.INT32,
    )
    page_table_v.set_uid(PAGE_TABLE_V_UID)

    o, _ = graph.sdpa(
        q=q,
        k=k,
        v=v,
        is_inference=True,
        attn_scale=0.08838834765,
        use_padding_mask=True,
        seq_len_q=seqlen_q,
        seq_len_kv=seqlen_kv,
        use_causal_mask=False,
        paged_attention_k_table=page_table_k,
        paged_attention_v_table=page_table_v,
        paged_attention_max_seq_len_kv=max_seq_len,
    )
    o.set_name("o")
    o.set_output(True)
    o.set_dim(q_dims)
    o.set_stride(q_strides)
    o.set_ragged_offset(ragged_offset)
    o.set_uid(O_UID)

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A])
    graph.check_support()
    graph.build_plans()

    return graph

def benchmark_highly_skewed():
    """Represents a scenario where one sequence has 256 input tokens while all the other have only 1"""
    token_count = 256 + 255
    batch_size = 256
    num_heads = 32
    num_kv_heads = 8
    head_size = 128
    block_size = 16
    num_blocks = 32768
    max_input_len = 256
    scale = 0.08838834765

    max_seq_len = 8192
    max_block_per_seq = max_seq_len // block_size

    seq_len = 2048

    cudnn_handle = cudnn.create_handle()

    graph = build_graph(batch_size, num_heads, num_kv_heads, head_size, block_size, num_blocks, max_input_len, max_seq_len)

    query = torch.randn(token_count, num_heads, head_size, dtype=torch.bfloat16)
    o = torch.zeros_like(query)
    key_cache = torch.randn(num_blocks, num_kv_heads, head_size, block_size, dtype=torch.bfloat16)
    value_cache = torch.randn(num_blocks, num_kv_heads, head_size, block_size, dtype=torch.bfloat16)

    values = torch.arange(0, num_blocks, dtype=torch.long)
    values = values[torch.randperm(num_blocks)]

    block_tables = torch.zeros((batch_size, max_block_per_seq), dtype=torch.int32)
    for i in range(batch_size):
        block_tables[i][0:seq_len//block_size] = torch.randint(0, num_blocks, (seq_len // block_size,), dtype=torch.int32)

    query_lens = [256] + [1 for i in range(batch_size - 1)]
    context_lens = [seq_len for query_len in query_lens]
    seqlen_q = torch.tensor(query_lens, dtype=torch.long)
    seqlen_kv = torch.tensor([a + b for a, b in zip(query_lens, context_lens)], dtype=torch.long)
    start_loc = torch.cumsum(torch.tensor([0] + query_lens, dtype=torch.long), dim=0) * num_heads * head_size

    variant_pack = {
        RAGGED_UID: start_loc,
        Q_UID: query,
        K_UID: key_cache,
        V_UID: value_cache,
        PAGE_TABLE_K_UID: block_tables,
        PAGE_TABLE_V_UID: block_tables,
        O_UID: o,
        SEQLEN_Q_UID: seqlen_q,
        SEQLEN_KV_UID: seqlen_kv,
    }

    start_time = time.time()
    ITERATIONS = 3000
    for _ in range(ITERATIONS):
        graph.execute(variant_pack, 0)
    torch.cuda.synchronize()
    end_time = time.time()

    print("Highly skewed scenario")
    print(f"cudnn Time: {(end_time - start_time)*1000:.2f} ms")
    print(f"cudnn Time per invocation: {((end_time - start_time) / ITERATIONS)*1000:.2f} ms\n")

    cudnn.destroy_handle(cudnn_handle)

def benchmark_decode_only():
    """Represents a scenario where all the sequence have 1 input token"""
    token_count = 256
    batch_size = 256
    num_heads = 32
    num_kv_heads = 8
    head_size = 128
    block_size = 16
    num_blocks = 32768
    max_input_len = 1
    scale = 0.08838834765

    max_seq_len = 8192
    max_block_per_seq = max_seq_len // block_size

    seq_len = 2048

    cudnn_handle = cudnn.create_handle()

    graph = build_graph(batch_size, num_heads, num_kv_heads, head_size, block_size, num_blocks, max_input_len, max_seq_len)

    query = torch.randn(token_count, num_heads, head_size, dtype=torch.bfloat16)
    o = torch.zeros_like(query)
    key_cache = torch.randn(num_blocks, num_kv_heads, head_size, block_size, dtype=torch.bfloat16)
    value_cache = torch.randn(num_blocks, num_kv_heads, head_size, block_size, dtype=torch.bfloat16)

    values = torch.arange(0, num_blocks, dtype=torch.long)
    values = values[torch.randperm(num_blocks)]

    block_tables = torch.zeros((batch_size, max_block_per_seq), dtype=torch.int32)
    for i in range(batch_size):
        block_tables[i][0:seq_len//block_size] = torch.randint(0, num_blocks, (seq_len // block_size,), dtype=torch.int32)

    query_lens = [1 for _ in range(batch_size)]
    context_lens = [seq_len for query_len in query_lens]
    seqlen_q = torch.tensor(query_lens, dtype=torch.long)
    seqlen_kv = torch.tensor([a + b for a, b in zip(query_lens, context_lens)], dtype=torch.long)
    start_loc = torch.cumsum(torch.tensor([0] + query_lens, dtype=torch.long), dim=0) * num_heads * head_size

    variant_pack = {
        RAGGED_UID: start_loc,
        Q_UID: query,
        K_UID: key_cache,
        V_UID: value_cache,
        PAGE_TABLE_K_UID: block_tables,
        PAGE_TABLE_V_UID: block_tables,
        O_UID: o,
        SEQLEN_Q_UID: seqlen_q,
        SEQLEN_KV_UID: seqlen_kv,
    }

    start_time = time.time()
    ITERATIONS = 3000
    for _ in range(ITERATIONS):
        graph.execute(variant_pack, 0)
    torch.cuda.synchronize()
    end_time = time.time()

    print("Decode only scenario")
    print(f"cudnn Time: {(end_time - start_time)*1000:.2f} ms")
    print(f"cudnn Time per invocation: {((end_time - start_time) / ITERATIONS)*1000:.2f} ms\n")

    cudnn.destroy_handle(cudnn_handle)

def benchmark_prefill_only():
    """Represents a scenario where all the sequence have 256 input tokens"""
    token_count = 256 * 256
    batch_size = 256
    num_heads = 32
    num_kv_heads = 8
    head_size = 128
    block_size = 16
    num_blocks = 32768
    max_input_len = 256
    scale = 0.08838834765

    max_seq_len = 8192
    max_block_per_seq = max_seq_len // block_size

    seq_len = 2048

    cudnn_handle = cudnn.create_handle()

    graph = build_graph(batch_size, num_heads, num_kv_heads, head_size, block_size, num_blocks, max_input_len, max_seq_len)

    query = torch.randn(token_count, num_heads, head_size, dtype=torch.bfloat16)
    o = torch.zeros_like(query)
    key_cache = torch.randn(num_blocks, num_kv_heads, head_size, block_size, dtype=torch.bfloat16)
    value_cache = torch.randn(num_blocks, num_kv_heads, head_size, block_size, dtype=torch.bfloat16)

    values = torch.arange(0, num_blocks, dtype=torch.long)
    values = values[torch.randperm(num_blocks)]

    block_tables = torch.zeros((batch_size, max_block_per_seq), dtype=torch.int32)
    for i in range(batch_size):
        block_tables[i][0:seq_len//block_size] = torch.randint(0, num_blocks, (seq_len // block_size,), dtype=torch.int32)

    query_lens = [256 for _ in range(batch_size)]
    context_lens = [seq_len for query_len in query_lens]
    seqlen_q = torch.tensor(query_lens, dtype=torch.long)
    seqlen_kv = torch.tensor([a + b for a, b in zip(query_lens, context_lens)], dtype=torch.long)
    start_loc = torch.cumsum(torch.tensor([0] + query_lens, dtype=torch.long), dim=0) * num_heads * head_size

    variant_pack = {
        RAGGED_UID: start_loc,
        Q_UID: query,
        K_UID: key_cache,
        V_UID: value_cache,
        PAGE_TABLE_K_UID: block_tables,
        PAGE_TABLE_V_UID: block_tables,
        O_UID: o,
        SEQLEN_Q_UID: seqlen_q,
        SEQLEN_KV_UID: seqlen_kv,
    }

    start_time = time.time()
    ITERATIONS = 3000
    for _ in range(ITERATIONS):
        graph.execute(variant_pack, 0)
    torch.cuda.synchronize()
    end_time = time.time()

    print("Prefill only scenario")
    print(f"cudnn Time: {(end_time - start_time)*1000:.2f} ms")
    print(f"cudnn Time per invocation: {((end_time - start_time) / ITERATIONS)*1000:.2f} ms\n")

    cudnn.destroy_handle(cudnn_handle)

if __name__ == '__main__':
    seed_everything(0)
    torch.set_default_device("cuda")

    benchmark_decode_only()
    benchmark_prefill_only()
    benchmark_highly_skewed()
