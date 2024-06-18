import json
import os
import re
from typing import Dict, Optional, Union, List
import faiss
import json5
import numpy as np
import pandas as pd

from qwen_agent.tools.base import BaseTool, register_tool

ROOT_RESOURCE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'agents/resource')


@register_tool('faq_embedding')
class GetFAQ(BaseTool):

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        self.recall_set, self.origin_data = self._build_recall_set(ROOT_RESOURCE + "/相似问.xls")

        if os.path.exists(ROOT_RESOURCE + "/faq.index"):
            self.recall_embedding = faiss.read_index(ROOT_RESOURCE + "/faq.index")
        else:
            self.recall_embedding = self._build_index(self.recall_set)

    def call(self, params: Union[str, dict], **kwargs) -> List[Dict]:
        params = self._verify_json_format_args(params)
        query = params['query']
        query_embedding = self.embedding_model.encode(query, normalize_embeddings=True).reshape(1, -1)
        score, prd_index = self.recall_embedding.search(query_embedding.astype(np.float32), k=5)
        prd_index = prd_index[score > 0.6]
        if len(prd_index) != 0:
            candidate_query = self.recall_set.iloc[prd_index]["query"].tolist()
            deduplicate_standard_query = list(set([self.standard_similar_mapping[c] for c in candidate_query]))
            filter_faq = self.origin_data[self.origin_data["标准问"].isin(deduplicate_standard_query)].to_dict(
                orient="records")
            return filter_faq
        else:
            return []

    def _build_index(self, df: pd.DataFrame):
        import os

        from sentence_transformers import SentenceTransformer
        ROOT_RESOURCE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'agents/resource')

        self.model_path = ROOT_RESOURCE + "/acge_text_embedding"
        self.embedding_model = SentenceTransformer(self.model_path)
        recall_embedding = self.embedding_model.encode(df["query"].tolist(), normalize_embeddings=True)
        dim = recall_embedding.shape[-1]
        # 根据嵌入向量的维度创建Faiss索引，使用内积作为度量标准
        faiss_index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
        # 将嵌入向量转换为float32类型，以满足Faiss的要求
        recall_embedding = recall_embedding.astype(np.float32)
        # 训练Faiss索引，以便其可以接受和处理嵌入向量
        faiss_index.train(recall_embedding)
        # 将所有嵌入向量添加到索引中
        faiss_index.add(recall_embedding)
        # 返回构建好的Faiss索引
        faiss.write_index(faiss_index, ROOT_RESOURCE + "/faq.index")
        return faiss_index

    def _build_recall_set(self, file_path):
        df = pd.read_excel(file_path)
        self.standard_similar_mapping = {}
        # 创建一个空的DataFrame来存储拆分后的数据
        new_df = pd.DataFrame(columns=['query', 'answer'])

        # 遍历每一行，拆分第一列的相似问，并添加到新的DataFrame中
        for index, row in df.iterrows():
            cleaned_similar_questions = re.sub(r"[\[\]'‘’“”]", '', row['相似问'])
            # 根据逗号拆分
            similar_questions = cleaned_similar_questions.split('，')
            standard_question, answer = row['标准问'], row["答案"]
            new_row = pd.DataFrame({'query': standard_question, 'answer': answer}, index=[0])
            new_df = pd.concat([new_df, new_row], ignore_index=True)
            self.standard_similar_mapping[standard_question] = standard_question
            for question in similar_questions:
                self.standard_similar_mapping[question] = standard_question
                new_row = pd.DataFrame({'query': question, 'answer': answer}, index=[0])
                new_df = pd.concat([new_df, new_row], ignore_index=True)
        return new_df, df
