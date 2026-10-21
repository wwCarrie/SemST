import os

# 国内网络无法直连 huggingface.co，改用镜像站 hf-mirror.com
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 禁用 Xet 存储（Qwen3 权重存在 xethub.hf.co，镜像站无法代理，需走传统 LFS 下载）
os.environ["HF_HUB_DISABLE_XET"] = "1"

import numpy as np
import scanpy as sc
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

"""把每个细胞的“高表达基因列表”转换成一个向量
    1.从数据里挑出表达量最高的10个基因
    2.10个基因名字拼成一句话
    3.喂给大语言模型
    4.让模型读这句话，输出一个代表该细胞“生物学状态”的向量
    5.所有细胞的向量存成.npy文件，方便训练使用
"""

# batch_size:批量计算
def generate_llm_emb(llm_model_name, all_gene_texts, batch_size, device_for_precomputation,
                     prompt_system, prompt_user, max_length=256, npy_path=None):
    if not os.path.exists(npy_path):
        print(f"Starting LLM base embedding precomputation on device: {device_for_precomputation}")

        # 分词器，转换为一系列token
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
        llm_base = AutoModel.from_pretrained(llm_model_name, trust_remote_code=True)
        # 批量计算需要补齐，使具有相同的序列长度
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        for param in llm_base.parameters():  # Freeze LLM base
            param.requires_grad = False
        llm_base.eval().to(device_for_precomputation)

        llm_hidden_dim = llm_base.config.hidden_size # 输出向量的长度
        all_base_embs_list = []

        # tqdm:循环增加进度条;desc:设置进度条的说明文字
        for i in tqdm(range(0, len(all_gene_texts), batch_size), desc="Precomputing LLM base embeddings"):
            batch_texts = all_gene_texts[i: i + batch_size]
            prompts = [
                f"<|im_start|>system\n{prompt_system}<|im_end|>\n<|im_start|>user\n{prompt_user}{text}<|im_end|>\n<|im_start|>assistant\n"
                for text in batch_texts
            ] # {text}里是基因列表
            # 将提示词转换成张量，此时每个spot对应一组Token向量 -> 每个spot对应一个向量
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(
                device_for_precomputation)

            with torch.no_grad():
                outputs = llm_base(**inputs, output_hidden_states=False)

            # 取平均得到句向量，代表整句话(细胞)的embedding
            # - 8：当前批次的spot数量
            # - 100：每个输入的Token数量
            # - 2560：每个Token的隐藏向量维度
            last_hidden_states = outputs.last_hidden_state # 8*100*2560
            # 取出注意力掩码 8*100
            # 增加一个维度 8*100*1
            # 最后一个维度扩展到2560：8*100*2560
            attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(last_hidden_states.size())

            # 填充Token的向量置零
            masked_hidden_states = last_hidden_states * attention_mask
            # 沿Token维度求和 8*2560
            summed_hidden_states = masked_hidden_states.sum(dim=1) 
            num_non_padding_tokens = attention_mask.sum(dim=1)
            num_non_padding_tokens = torch.clamp(num_non_padding_tokens, min=1e-9)
            # 掩码平均池化
            pooled_output = summed_hidden_states / num_non_padding_tokens

            all_base_embs_list.append(pooled_output.cpu())

        del llm_base, tokenizer  # Free memory
        if torch.cuda.is_available() and device_for_precomputation.type == 'cuda':
            torch.cuda.empty_cache()

        # 拼接所有批次，假设2695个spot，每个spot向量维度2560
        # 2695*2560
        final_base_embeddings = torch.cat(all_base_embs_list, dim=0)
        print(
            f"Finished LLM base embedding precomputation. "
            f"Shape: {final_base_embeddings.shape}, "
            f"Dimension: {llm_hidden_dim}, "
            f"path: {npy_path}"
        )
        np.save(npy_path, final_base_embeddings)


if __name__ == '__main__':

    """
    Input:
        - .h5ad数据格式
    Scanpy读取后，返回一个AnnData对象
    - adata.X：基因表达矩阵。
    - adata.obs：spot 或细胞的元信息。
    - adata.var：基因的元信息。
    - adata.obsm：空间坐标等多维数据。
    """

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_TOP_GENES_LLM = 10 # 取前10个基因
    LLM_MAX_LENGTH = 2560
    LLM_PRECOMPUTATION_BATCH_SIZE = 8

    LLM_MODEL_NAME = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model", "Qwen3-4B")
    datasets = ['Mouse_Brain_Anterior']
    # datasets = ['Human_Breast_Cancer']
    for dataset_name in datasets:

        NPY_PATH = rf'data/new_npys/embeddings_{dataset_name}.npy'
        if not os.path.exists(os.path.dirname(NPY_PATH)):
            os.makedirs(os.path.dirname(NPY_PATH))

        if dataset_name in ['151507', '151508', '151509', '151510', '151669', '151670', '151671', '151672']:
            path = f"generate_data/DLPFC/{dataset_name}/data.h5ad"
        elif dataset_name == 'E1S1':
            path = f"generate_data/Embryo/{dataset_name}/data.h5ad"
        else:
            path = f"generate_data/{dataset_name}/data.h5ad"
        adata = sc.read_h5ad(path)

        # toarray:不是NumPy数组，则转为稠密矩阵。稠密矩阵需要为所有位置分配存储空间
        gene_expression_raw = adata.X.toarray() if isinstance(adata.X, np.ndarray) is False else adata.X
        # 提取表达量最高的10个索引（基因）
        top_n_gene_indices = np.argsort(gene_expression_raw, axis=1)[:, -N_TOP_GENES_LLM:] # 按表达量排序，取下标
        cell_gene_texts = []
        gene_names_array = adata.var_names.to_numpy()
        for i in range(adata.n_obs):
            # 找出下标对应的基因名字
            top_genes_for_cell = gene_names_array[top_n_gene_indices[i]]
            cell_gene_texts.append(", ".join(reversed(top_genes_for_cell))) # 反转后从高到低

        prompt_system = (
            'You are an expert in bioinformatics. Represent the biological state of a cell characterized '
            'by the following highly expressed genes. Focus on capturing the functional essence relevant '
            'for spatial domain identification. Output a dense vector representation.')

        prompt_user = 'Highly expressed genes: '

        generate_llm_emb(
            llm_model_name=LLM_MODEL_NAME,
            all_gene_texts=cell_gene_texts,
            batch_size=LLM_PRECOMPUTATION_BATCH_SIZE,
            device_for_precomputation=DEVICE,
            prompt_system=prompt_system,
            prompt_user=prompt_user,
            max_length=LLM_MAX_LENGTH,
            npy_path=NPY_PATH
        )
