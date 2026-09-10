"""
rag_retriever.py —— RAG 检索全链路
=================================================================
职责：文档加载 -> 分块 -> 向量化入库(Chroma) -> 混合检索 -> 重排。

对 Java 工程师的类比说明：
- 整个流程对应 Java 的"数据管道 Pipeline"：
  loader(读取) -> splitter(ETL 清洗拆分) -> 向量库(ES 式索引) -> 检索(查询)。
- Chroma 类似轻量版 Elasticsearch：本地持久化，集合(collection)等价于 index。
- 混合检索 = 多路召回 + 融合排序：
  · Dense(向量相似度)   -> 语义召回，等价于 ES 的 dense_vector kNN
  · Sparse(BM25 词频)   -> 关键词召回，等价于 ES 的 BM25 全文检索
  · RRF 融合            -> 多路结果合并排序（业界标准算法）
  · LLM 重排            -> 精排（cross-encoder 的精简替代，无需下载模型）
"""
import logging
import re
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

from . import config
from .trace import trace

logger = logging.getLogger("rag")


class RetrievedChunk:
    """一次检索命中的单个上下文块（轻量数据类）。"""

    __slots__ = ("score", "source", "text")

    def __init__(self, text: str, source: str, score: float) -> None:
        self.text = text
        self.source = source      # 来源文件名，用于"溯源引用"
        self.score = score        # 融合后的得分

    def to_context_str(self) -> str:
        """转成可直接拼进 Prompt 的文本，带来源标记便于溯源。"""
        return f"【来源：{self.source}】\n{self.text}"


def _tokenize(text: str) -> list[str]:
    """无依赖的中文分词近似：CJK 部分按双字(n-gram)切分，英文/数字按单词。

    说明：中文没有天然空格，BM25 需要 token 序列。这里用 bigram 兜底，
    工程上可替换为 jieba 分词（不改变调用方接口）。
    """
    tokens: list[str] = []

    # 提取连续的 CJK 汉字串
    for cjk_run in re.findall(r"[\u4e00-\u9fff]+", text):
        # 双字切分：例如 "营业收入" -> ["营业","业收","收入"]
        tokens.extend(cjk_run[i : i + 2] for i in range(len(cjk_run) - 1))
        # 单字也保留，避免过于稀疏
        tokens.extend(cjk_run)
    # 提取连续的英文/数字单词
    tokens.extend(re.findall(r"[A-Za-z0-9]+", text))
    return tokens


class FinancialRetriever:
    """RAG 检索器：负责索引构建与混合检索。

    使用方式：
        retriever = FinancialRetriever()
        retriever.ensure_index()      # 首次自动构建，之后增量加载
        chunks = await retriever.retrieve("华信科技 2024 年营收是多少？")
    """

    COLLECTION_NAME = "financial_docs"

    def __init__(self) -> None:
        # Embedding 与向量库的持久化目录都来自统一配置
        self.embeddings = config.get_embeddings()
        self._vectorstore: Chroma | None = None
        self._bm25: BM25Okapi | None = None
        self._bm25_docs: list[RetrievedChunk] = []

    # ------------------------------------------------------------------
    # 索引构建
    # ------------------------------------------------------------------
    @trace("index.build")
    def ensure_index(self) -> None:
        """确保向量索引存在；已存在则直接加载（增量复用，避免重复 embed）。

        原理：Chroma 集合的 count()>0 说明历史索引还在持久化目录里，
        此时直接复用；否则重新走"加载->分块->入库"全流程。
        """
        if self._vectorstore is not None:
            return

        # Chroma 传入 persist_directory 会自动使用本地持久化客户端
        self._vectorstore = Chroma(
            collection_name=self.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=config.CHROMA_PERSIST_DIR,
        )
        # 局部变量 + assert 收窄类型，消除静态检查的 Optional 访问告警
        vs = self._vectorstore
        assert vs is not None

        # 用公开 API get() 判断集合是否已有数据，替代私有属性 _collection.count()
        if vs.get(include=[], limit=1).get("ids"):
            logger.info("检测到已有向量索引，直接加载")
            self._refresh_bm25_from_store()
            return

        logger.info("首次构建索引：加载 %s 目录文档 ...", config.DATA_DIR)
        chunks = self._load_and_split()
        if not chunks:
            raise RuntimeError(f"目录 {config.DATA_DIR} 下没有可索引的文档")
        logger.info("分块完成，共 %s 块，开始向量化入库 ...", len(chunks))

        # 向量化 + 入库（一次批量操作）。
        # 注意：embed_documents 是同步阻塞调用，等价于 Java 中同步的 REST 批量请求；
        # 只在构建期执行一次，不影响线上检索路径的并发。
        vs.add_documents(chunks)
        logger.info("向量索引构建完成，持久化到 %s", config.CHROMA_PERSIST_DIR)

        self._refresh_bm25_from_store()

    def _load_and_split(self) -> list[Document]:
        """按扩展名分流加载 + 递归字符切分。

        - TXT 走 DirectoryLoader：等价于 Java 的"目录扫描器 + 自定义 Reader"，
          通过 glob 过滤文件类型，通过 loader_cls 指定解析器。
        - PDF 刻意不走 DirectoryLoader：其 loader_cls 的类型标注未包含
          PyPDFLoader（langchain_community 标注缺陷，运行时其实合法），
          直接逐文件构造 PyPDFLoader 可让类型检查完全干净、无需 ignore。
        - RecursiveCharacterTextSplitter 是 langchain 推荐的通用分块器：
          按优先分隔符递归切分（先段落、再句子、再字符），
          比固定长度硬切更符合自然语义边界。
        """
        data_dir = Path(config.DATA_DIR)
        if not data_dir.exists():
            raise RuntimeError(f"数据目录不存在：{data_dir}，请确认已放置文档")

        loaders: list[DirectoryLoader] = []
        if list(data_dir.glob("*.txt")):
            loaders.append(DirectoryLoader(
                str(data_dir), glob="*.txt", loader_cls=TextLoader,
            ))

        docs: list[Document] = []
        for loader in loaders:
            docs.extend(loader.load())

        # sorted 保证多文件加载顺序确定，索引重建结果可复现
        for pdf_path in sorted(data_dir.glob("*.pdf")):
            docs.extend(PyPDFLoader(str(pdf_path)).load())

        # chunk_size=500 / chunk_overlap=50：与 .env 配置一致
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,          # 按字符数切分（中文场景）
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )
        return splitter.split_documents(docs)

    def _refresh_bm25_from_store(self) -> None:
        """从向量库拉取全部文本，重建 BM25 稀疏索引（供混合检索）。"""
        assert self._vectorstore is not None
        raw = self._vectorstore.get(include=["documents", "metadatas"])
        texts: list[str] = raw.get("documents", []) or []
        metas: list[dict[str, Any]] = raw.get("metadatas", []) or []
        if not texts:
            return

        self._bm25_docs = [
            RetrievedChunk(
                text=t,
                source=str((m or {}).get("source", "unknown")),
                score=0.0,
            )
            for t, m in zip(texts, metas)
        ]
        # 预分词并构建 BM25 索引（O(N*len) 一次构建，查询 O(1)）
        self._bm25 = BM25Okapi([_tokenize(d.text) for d in self._bm25_docs])
        logger.info("BM25 稀疏索引就绪，共 %s 块", len(self._bm25_docs))

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    @trace("rag.retrieve_dense")
    async def retrieve_dense(self, query: str, k: int | None = None) -> list[RetrievedChunk]:
        """纯向量检索（基线）：仅 Dense 召回，不融合、不重排。

        用于评估实验中的"策略一"，对比混合检索与重排带来的增益。
        """
        k = k or config.RETRIEVAL_TOP_K
        self.ensure_index()
        assert self._vectorstore is not None

        hits = self._vectorstore.similarity_search_with_score(query, k=k)
        return [
            RetrievedChunk(doc.page_content, doc.metadata.get("source", ""), score)
            for doc, score in hits
        ]

    @trace("rag.retrieve")
    async def retrieve(self, query: str, k: int | None = None,
                       rerank: bool | None = None) -> list[RetrievedChunk]:
        """混合检索：Dense(BM25) 多路召回 -> RRF 融合 -> LLM 重排。

        参数：
            query: 用户问题原文
            k:     最终返回的上下文块数，默认取配置 RETRIEVAL_TOP_K
            rerank: 是否启用 LLM 重排；None 表示跟随 config.ENABLE_RE_RANK，
                    评估实验可显式传入 True/False 做对比。
        """
        k = k or config.RETRIEVAL_TOP_K
        self.ensure_index()
        assert self._vectorstore is not None and self._bm25 is not None

        # 多路召回时多取一些候选，供融合与重排筛选
        candidate_k = max(k * 2, k + 4)

        # ---- 第 1 路：Dense 向量检索（语义相似） ----
        # similarity_search_with_score 返回 [(Document, distance)]，按距离升序
        # （越相似越靠前）。RRF 融合只依赖"排名"而非原始分数，因此无需关心
        # 距离量纲；相比 relevance_scores，避免了 embedding 相似度越界触发的
        # "Relevance scores must be between 0 and 1" 警告噪音。
        dense_hits = self._vectorstore.similarity_search_with_score(
            query, k=candidate_k
        )

        # ---- 第 2 路：Sparse BM25 检索（关键词精确命中） ----
        bm25_scores = self._bm25.get_scores(_tokenize(query))
        # 按 BM25 得分降序取 top candidate_k
        top_idx = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[:candidate_k]

        # ---- RRF（Reciprocal Rank Fusion）融合两路结果 ----
        # 核心公式：score(chunk) = Σ 1/(k + rank_i)，k 通常取 60。
        # 只依赖"排名"而非原始分数，天然消除两路量纲差异（对应 Java 中的
        # 排序融合算法，如搜索结果的多路 merge）。
        k = 60
        fused: dict[str, RetrievedChunk] = {}

        for rank, (doc, _score) in enumerate(dense_hits):
            key = doc.page_content
            fused.setdefault(key, RetrievedChunk(doc.page_content, doc.metadata.get("source", ""), 0.0))
            fused[key].score += 1.0 / (k + rank + 1)

        for rank, idx in enumerate(top_idx):
            chunk = self._bm25_docs[idx]
            key = chunk.text
            if key not in fused:
                fused[key] = RetrievedChunk(chunk.text, chunk.source, 0.0)
            fused[key].score += 1.0 / (k + rank + 1)
        fused_list = sorted(fused.values(), key=lambda c: c.score, reverse=True)

        # ---- LLM 重排（精排，可选） ----
        # 对融合后的 top 候选用 LLM 逐条打分，纠正"词面相关但语义不相关"的噪声。
        # 零额外模型依赖；生产环境可替换为 BGE-Reranker 交叉编码器，接口一致。
        if (config.ENABLE_RE_RANK if rerank is None else rerank) and fused_list:
            fused_list = await self._llm_rerank(query, fused_list, k)

        result = fused_list[:k]
        logger.info("[RAG] query=%s -> 召回 %s 块，命中来源=%s",
                    query, len(result), {c.source for c in result})
        return result

    async def _llm_rerank(self, query: str, candidates: list[RetrievedChunk],
                          k: int) -> list[RetrievedChunk]:
        """用 LLM 对候选块打分重排。

        实现要点：
        1) 一次性把所有候选给 LLM，要求按"最相关"到"最不相关"排序并评分；
        2) 用正则从回答中解析出每条 score，解析失败时保守降级（保持原顺序），
           绝不因重排失败中断检索主链路（熔断降级思想）。
        """
        llm = config.get_llm()

        numbered = "\n".join(
            f"{i + 1}. {c.text[:120]}" for i, c in enumerate(candidates)
        )
        prompt = (
            f"你是一个检索质量打分器。给定用户问题与候选文档片段，"
            f"请按相关性从高到低对候选重新排序，并给每个片段打 0~10 分。\n"
            f"严格输出格式：每行 `序号|分数`，如 `3|9.5`。\n\n"
            f"用户问题：{query}\n\n候选片段：\n{numbered}"
        )

        try:
            resp = await llm.ainvoke(prompt)
            # 逐行解析 "序号|分数"
            pattern = re.compile(r"^\s*(\d+)\s*[|:]\s*([0-9]+(?:\.[0-9]+)?)\s*$")
            order: list[tuple[int, float]] = []
            for line in str(resp.content).splitlines():
                m = pattern.match(line)
                if m:
                    order.append((int(m.group(1)) - 1, float(m.group(2))))
            if len(order) < 2:
                return candidates  # 解析失败，降级为原顺序
            # 按分数降序重排，超出范围的序号忽略
            reranked = []
            for idx, _score in sorted(order, key=lambda t: t[1], reverse=True):
                if 0 <= idx < len(candidates) and candidates[idx] not in reranked:
                    reranked.append(candidates[idx])
            for c in candidates:
                if c not in reranked:
                    reranked.append(c)
            return reranked
        except Exception as exc:  # noqa: BLE001 - 重排失败刻意兜底降级，不让主链路中断
            logger.warning("LLM 重排失败，降级为 RRF 原顺序: %s", exc)
            return candidates
