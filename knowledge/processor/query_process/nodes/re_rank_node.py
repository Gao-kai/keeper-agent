import logging
from typing import List, Literal, Tuple, TypedDict

from knowledge.processor.query_process.base import BaseNode
from knowledge.processor.query_process.config import QueryConfig, get_query_config
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.bge_reranker_base_model import get_bge_reranker_model
from knowledge.utils.log_config import setup_logging


class ReRankDocItemModel(TypedDict):
    """重排序阶段统一的文档结构。

    将 RRF 融合结果与 Web 搜索结果归一化成同一种结构，
    便于后续统一送入 Reranker 打分、断崖检测截断。

    Attributes:
        content: 文档正文，Reranker 实际用于计算相关性的文本。
            Web 来源取 snippet，取不到时回退到 content。
        source: 文档来源，用于区分召回链路。
            "web" 表示 Web 搜索结果，"rrf" 表示本地知识库经 RRF 融合后的结果。
            写成 Literal 而非 "web" | "rrf"：| 只对类型对象生效，
            对字符串实例会抛 TypeError。
        chunk_id: 本地知识库切片 ID，Web 结果没有该字段，为 None。
            下游 RRF 依赖它做去重投票，Web 文档因无 ID 不参与投票。
        title: 文档标题。Web 来源取网页标题，
            本地来源取 file_title（缺失时为空串）。
        url: 文档链接，用于答案溯源时展示引用来源。
            本地知识库切片为空串。
    """

    content: str
    source: Literal["web", "rrf"]
    chunk_id: str | None
    title: str
    url: str


class ReRankScoredDocItemModel(ReRankDocItemModel):
    """重排序后的文档结构，在 ReRankDocItemModel 基础上增加相关性分数。

    单独拆出来而不是直接加在父类，是因为 score 是 Reranker 打分之后
    才有的：构造候选集阶段没有该键，用类型把两个阶段区分开，
    静态检查器可以拦住"未打分就传给断崖检测"这类错误。

    Attributes:
        score: Reranker 计算出的相关性分数，降序排列后用于断崖检测。
            取 None 表示 Reranker 降级（模型加载失败 / 显存不足 /
            输入超长），此时下游不参与截断判断，直接跳过。
            注意 compute_score 返回 numpy.float32，需 float() 转换，
            否则 json 序列化会报 float32 不可序列化。
    """

    score: float | None


class ReRankNode(BaseNode):
    """重排序节点。

    位于 RRF 融合与 Web 搜索之后、答案生成之前，负责把多路召回的候选文档
    交给 Reranker 精排，并用断崖检测动态决定最终保留多少条。

    处理流程：
        1. normalize_doc_items: 把 RRF 结果和 Web 搜索结果归一化成同一种结构
        2. rerank_docs: 用 BGE Reranker 对 (问题, 文档) 逐对打分并倒序排列
        3. cliff_cut_docs: 断崖检测动态截断，替代固定 TopK
    """

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """执行重排序并写回 state。

        Args:
            state: 查询流程图状态。读取 rewritten_query / original_query、
                rrf_chunks、web_search_docs；写回 reranked_docs。

        Returns:
            更新了 reranked_docs 的 state。
        """
        config = get_query_config()
        # 优先用改写后的问题（语义更完整）；改写失败时回退原始问题
        question = state.get("rewritten_query") or state.get("original_query", "")

        # 1. 统一RRF融合排序后的chunk和通过Web搜索的文档为相同数据格式
        doc_items: List[ReRankDocItemModel] = self.normalize_doc_items(state)

        # 2. 调用re-rank重排序模型进行得分计算
        ranked_doc_items: List[ReRankScoredDocItemModel] = self.rerank_docs(
            doc_items, question
        )

        # 3. 执行动态Top-K断崖检测
        top_k_docs = self.cliff_cut_docs(ranked_doc_items, config)
        self.logger.info(f"重排序完成: {len(ranked_doc_items)} → {len(top_k_docs)}")

        # 4. 更新state
        state["reranked_docs"] = top_k_docs
        return state

    def normalize_doc_items(self, state: QueryGraphState) -> List[ReRankDocItemModel]:
        """把 RRF 结果与 Web 搜索结果归一化成 ReRankDocItemModel。

        两类数据源字段命名不同，这里统一成 content / source / chunk_id /
        title / url 五个字段，让下游 Reranker 无需关心文档来源。

        为什么网络搜索结果不参与 RRF 而在这里才加入：RRF 靠 chunk_id 投票，
        网络文档没有 chunk_id 投不了；Reranker 靠语义打分，两者能同台竞争。

        Args:
            state: 查询流程图状态，读取 rrf_chunks 和 web_search_docs。

        Returns:
            归一化后的文档列表；content 为空的文档已被过滤。
        """
        web_search_docs = state.get("web_search_docs", [])
        rrf_chunks = state.get("rrf_chunks", [])

        doc_items: List[ReRankDocItemModel] = []

        for web_search_doc_item in web_search_docs:
            if not isinstance(web_search_doc_item, dict):
                continue

            # Web 搜索结果优先用 snippet（摘要更贴近问题），取不到再回退 content
            snippet = web_search_doc_item.get("snippet", "").strip()
            content = web_search_doc_item.get("content", "").strip()
            item_content = snippet or content
            title = web_search_doc_item.get("title", "").strip()
            url = web_search_doc_item.get("url", "").strip()

            # 正文为空的文档无法参与语义打分，直接丢弃，避免浪费 reranker 算力
            if not item_content:
                continue

            doc_items.append(
                ReRankDocItemModel(
                    content=item_content,
                    source="web",
                    # 网络文档没有本地切片 ID，置 None：
                    # 下游若需去重投票，用 `if chunk_id` 即可区分
                    chunk_id=None,
                    title=title,
                    url=url,
                )
            )

        for rrf_doc_item in rrf_chunks:
            if not isinstance(rrf_doc_item, dict):
                continue

            content = rrf_doc_item.get("content", "").strip()
            # 本地切片的标题字段叫 file_title，与 Web 的 title 对齐
            title = rrf_doc_item.get("file_title", "").strip()
            url = rrf_doc_item.get("url", "").strip()

            if not content:
                continue

            doc_items.append(
                ReRankDocItemModel(
                    content=content,
                    source="rrf",
                    # chunk_id 可能是 int（如 468308814313816318），统一转成 str；
                    # 缺失或为 None 时保持 None，不能直接 str(None) 否则会得到 "None" 字符串
                    chunk_id=self._normalize_chunk_id(rrf_doc_item.get("chunk_id")),
                    title=title,
                    url=url,
                )
            )

        self.logger.info(f"合并文档: {len(doc_items)} 篇")

        return doc_items

    @staticmethod
    def _normalize_chunk_id(raw_chunk_id) -> str | None:
        """把切片 ID 统一转成字符串。

        Args:
            raw_chunk_id: 原始切片 ID，可能是 int / str / None。

        Returns:
            去空格后的字符串；原始值为空或 None 时返回 None。
        """
        if raw_chunk_id is None or raw_chunk_id == "":
            return None
        chunk_id = str(raw_chunk_id).strip()
        # 防止 str(None) 这种已转成字符串的情况
        return chunk_id if chunk_id and chunk_id != "None" else None

    def rerank_docs(
        self, doc_items: List[ReRankDocItemModel], question: str
    ) -> List[ReRankScoredDocItemModel]:
        """
        重排序模型
        Args:
            doc_items: 归一化后的候选文档（本地 RRF + Web 搜索）
            question: 用户问题（优先用改写后的 rewritten_query）

        Returns:
            按 reranker 得分倒序排列的文档列表。
            降级（模型未加载 / 推理异常）时返回原序且 score 为 None，
            下游断崖检测遇到 None 会跳过该对比，不会报错。
        """

        def _degrade_item_docs() -> List[ReRankScoredDocItemModel]:
            """降级：保持原序，score 置空。"""
            return [
                ReRankScoredDocItemModel(
                    content=item["content"],
                    source=item["source"],
                    chunk_id=item["chunk_id"],
                    title=item["title"],
                    url=item["url"],
                    score=None,
                )
                for item in doc_items
            ]

        # 没有候选或没有问题时无需打分，直接返回空列表，避免把空列表喂给模型
        if not doc_items or not question:
            return []

        try:
            bge_reranker_model = get_bge_reranker_model()
            if bge_reranker_model is None:
                return _degrade_item_docs()

            # 构建排序模型 compute_score 所需的 (问题, 文档) 参数对列表。
            # 用元组而非列表：源码签名是 List[Tuple[str, str]]，
            # 元组语义上表示"固定两个元素的配对"，更贴合问题-文档对的含义。
            re_rank_pairs: List[Tuple[str, str]] = [
                (question, doc_item["content"]) for doc_item in doc_items
            ]

            # 执行计算 计算每一对Question-Content的得分。
            # 注意返回的是 List[float]（源码里 all_scores 是 Python list），
            # 不是 numpy.ndarray；元素为 numpy.float32，需 float() 转换。
            re_rank_scores: List[float] = bge_reranker_model.compute_score(
                re_rank_pairs
            )

            # 收集结果
            # 显式构造 TypedDict，而不是 {**doc_item, "score": ...}：
            # 字典解包在静态检查里只会推导出 dict[str, object]，无法匹配 TypedDict 类型
            results: List[ReRankScoredDocItemModel] = []
            for doc_item, re_rank_score_by_doc_content in zip(
                doc_items, re_rank_scores, strict=True
            ):
                results.append(
                    ReRankScoredDocItemModel(
                        content=doc_item["content"],
                        source=doc_item["source"],
                        chunk_id=doc_item["chunk_id"],
                        title=doc_item["title"],
                        url=doc_item["url"],
                        # compute_score 返回 numpy.float32，转 float 才能 json 序列化
                        score=float(re_rank_score_by_doc_content),
                    )
                )

            # 按照得分倒序排序。
            # 用 item["score"] or 兜底：正常路径 score 必然是 float，
            # 但保险起见让 None 排在最后（0.0 等价于排在末尾），避免 TypeError
            return sorted(
                results,
                key=lambda item: item["score"] if item["score"] is not None else 0.0,
                reverse=True,
            )

        except Exception as e:
            self.logger.error(f"执行重排序re-rank失败:{e}")
            self.logger.error(f"降级为按照原序列排序返回")
            return _degrade_item_docs()

    def cliff_cut_docs(
        self, ranked_doc_items: List[ReRankScoredDocItemModel], config: QueryConfig
    ) -> List[ReRankScoredDocItemModel]:
        """断崖检测动态截断，替代固定 TopK。

        固定 TopK 的两个问题：真实相关的只有 3 条时会混入噪声；
        前 7 条都相关时会丢失有价值的第 6、7 条。这里改为寻找得分
        "断崖式下跌"的位置截断，最少保留 min_topk 条（保底），
        最多保留 max_topk 条（封顶）。

        为什么需要两个阈值：
            绝对阈值调高了低分区漏检，调低了高分区误杀。
            高分区（如 8.0→7.2）绝对差距大但比例小，靠 abs_gap 抓；
            低分区（如 0.8→0.5）绝对差距小但比例大，靠 rel_gap 抓。
            两者用 or 连接，互相补位。

        Args:
            ranked_doc_items: 按 reranker 得分倒序排列的文档列表。
            config: 查询流程配置，读取 rerank_max_topk / rerank_min_topk /
                rerank_gap_abs / rerank_gap_ratio。

        Returns:
            截断后的文档列表；质量断崖之后的低分文档被丢弃。
        """
        if not ranked_doc_items:
            return []

        # 最多不能超过文档总数（候选不足 max_topk 时），最少也不能超过最大值
        upper_bound = min(config.rerank_max_topk, len(ranked_doc_items))
        lower_bound = min(config.rerank_min_topk, upper_bound)

        # 默认取到上限：没检测到断崖就说明质量平稳，全取
        # （注意不能初始化成 lower_bound，否则未检出断崖时会丢掉 4~max_topk 条）
        cut_off_index = upper_bound

        # 从 lower_bound 之后的第一对开始检查：
        # 前 lower_bound 条是保底量，无论得分如何都保留，不参与断崖判断。
        # 检查区间是 [lower_bound-1, upper_bound-1)，即比较第 i 条与第 i+1 条。
        for index in range(lower_bound - 1, upper_bound - 1):
            curr_score = ranked_doc_items[index].get("score")
            next_score = ranked_doc_items[index + 1].get("score")

            # 分数为空说明 reranker 降级，无法比较，跳过该对
            if curr_score is None or next_score is None:
                continue

            # 绝对差距：捕捉高分区的大幅下跌
            abs_gap = curr_score - next_score
            # 相对差距：捕捉低分区的比例性下跌。
            # 分母加 1e-6 有两层作用：防止 curr_score 为 0 时除零；
            # 分数可能是负数，用 abs() 保证分母为正、比例方向正确。
            rel_gap = abs_gap / (abs(curr_score) + 1e-6)

            # 任一条件满足即为断崖，立即截断（用 or 而非 and，否则明显的断崖会漏检）
            if abs_gap >= config.rerank_gap_abs or rel_gap >= config.rerank_gap_ratio:
                cut_off_index = index + 1
                self.logger.debug(
                    f"断崖检测: 位置 {index + 1}, "
                    f"绝对差距={abs_gap:.4f}, 相对差距={rel_gap:.4f}"
                )
                break

        return ranked_doc_items[:cut_off_index]


if __name__ == "__main__":
    setup_logging()
    print("开始测试Re Ranker重排序节点")

    web_search_docs = [
        {
            "snippet": "今日重点一览|科技圈大事速读 🔥今日重点一览|科技圈大事速读 1️⃣ Redmi 15 渲染图曝光,搭载骁龙 6s Gen 3 2️⃣ Neuralink 联合研发 AI 仿生眼,助盲人复明 3️⃣ GPT-5 曝光:代号“龙虾”,编程能力惊艳 4️⃣ 马斯克放话:年底前推特斯拉平价车型 5️⃣ 国首款 6nm 显卡亮相,《黑神话:悟空》4K流畅跑 6️⃣ 小米空调新品2026将上市,卢伟冰自信拉满 7️⃣ B站国际版将于8月5日关闭,第三方客户端疑被风控 8️⃣ Meta新任AI科学家赵晟佳正式官宣 9️⃣ 苹果大连Apple Store关闭,深圳开新店 🔟 特斯拉股价重挫,“自动驾驶梦想”遭质疑 🚀 科技硬核新品扎堆亮相,国产力量迎来爆发 小米Redmi 15 手机渲染图曝光,提供黑、紫、淡金三色选择,配备 6.9 英寸 FHD+ 144Hz 屏幕、骁龙 6s Gen 3 芯片、7000mAh 电池,预装 HyperOS,起售价仅约合 1000 元人民币,性价比拉满。 国产显卡厂商砺算科技发布首批 6nm 消费级显卡 7G106,定位对标 RTX 4060,实机演示《黑神话:悟空》4K 高画质下帧率超 70,标志着国产 GPU 技术迈出关键一步。、 同时,宇树科技推出双足机器人 UnitreeR1,售价仅 3.99 万元,可完成倒立、翻跟头等动作,未来或广泛应用于工业与家用服务场景。 小米也不甘示弱,卢伟冰晒出小米空调内部结构,透露 2026 年新品将具备极强竞争力,或在能效与智能交互方面大幅升级。 🧠 AI 与脑机接口齐头并进,Neuralink、GPT-5 引爆话题 Neuralink 正联合研究团队开发 AI 智能仿生眼“Blindsight”,计划2030年上市,可辅助盲人识别面孔、导航、阅读。 OpenAI 的 GPT-5 模型代号“龙虾”悄然亮相,在 WebDev Arena 编程测试中远超当前主流模型,表现亮眼,表明其在自然语言推理、编程、写作等方向全面提升。 Meta 则官宣前 OpenAI 核心成员赵晟佳为 MSL 首席科学家,目标直指 AI 推理领域核心技术突破,配合 Prometheus 超级算力平台,意图与 OpenAI 一较高下。 💰 特斯拉频传利空,销量疲软 股价重挫 特斯拉二季度利润下滑 16%,股价跌超 20%。尽管马斯克称年底将基于 Model Y 推出亲民车型,寄望于自动驾驶出租车与机器人业务,但投资者似乎不再“买账”。 辅助驾驶安全问题也屡见报道:沪渝高速发生撞护栏事故,司机因过度依赖 AI 驾驶未及时介入。交警提醒,辅助驾驶非自动驾驶,仍需人工干预。 🧭 中国科技出海起伏交错,B站、Apple Store调布局 Bilibili 国际版将于 8 月 5 日关停,多款第三方客户端疑似触发风控机制。内 5️⃣",
            "title": "今日重点一览|科技圈大事速读",
            "url": "https://www.bilibili.com/read/cv42466987",
        },
        {
            "snippet": "中国科技圈又发生了一件大事,跟每个人息息相关,只不过大家都还没意识到。那就是比亚迪和美的联手了,未来要在智能生活领域掀起一场巨大变革。 比亚迪和美的,一个是全球新能源汽车领导者,一个是全球领先的智能家居科技公司,他们之间达成合作,这含金量,懂得人都懂。别的不说,肯定会整合双方在智能汽车、智能家居及AIoT等核心优势,打破车与家的传统边界,共同打造“人-车-家”智慧新生态! 比亚迪大家都比较熟悉,不仅技术牛、车卖的好,还在多个尖端领域广泛布局。和美的的合作,就是比亚迪开放合作的一部分,聚焦于AI智能体(Agent)层面的协同创新。也就是说未来智能汽车与家庭环境智能管家之间,或许很可能会实现数据互通与协同决策。就好比为车与家搭建了一座畅通无阻的信息桥梁,让两者能够无缝对话。 从智能家电到智能家居设备,再到IoT车载产品等全品类产品都将逐步接入,构建全面无感互联。“车控家”与“家控车”将实现无缝衔接。或许你在下班路上,只需在车上轻轻一点,家中的空调就会提前开启,调节到适宜的温度。或许当你准备出门时,在家中一键就能完成“远程备车”,检查车辆状态、预热发动机等,从容应对出行需求。 新能源汽车时代,中国车企在不断努力和实现技术突破。随着这些科幻电影一样的场面走进现实,中国汽车产业将真正站在全球的前列。据说腾势N8L、2026款夏、汉L及唐L等车型上率先“上车”,那就让我们一起期待吧!",
            "title": "中国科技圈又发生了一件大事,跟每个人息息相关,只不过大家都还没意识到。那就是比亚迪和美的联手了,未来要在智能生活领域掀起一场巨大变革。",
            "url": "https://weibo.com/7420350127/QflByec7s",
        },
        {
            "snippet": "聚焦科技圈:今天有哪些大事?快速一览 今日要闻最新消息 专题目录 今日要闻辛国斌:机器人产业迎来了创新发展、升级换代的重要机遇界面新闻腾讯24年增长断片,马化腾还能靠谁单骑救主? 4创事记人工审核用户照片?百度网盘:系谣言!网友凌晨发文回应 3每日经济新闻网拼多多正筹建跨境电商平台:密集挖角shein员工,0佣金招商入驻 46界面新闻郭明錤:预计apple watch 8从越南出货的比重会提升到60-70% 7新浪科技展开更多最新消息消息称特斯拉最快下周宣布在墨西哥建厂 2新浪科技台积电计划2024年在亚利桑那州工厂生产4纳米芯片财联社哪吒汽车11月交付量15072台 同比增长51%财联社吉利汽车:极氪11月交付汽车共1.1万部,同比增长约447%界面新闻集邦咨询:oled手机渗透率2023年将达50.8%财联社展开更多全部评论 28万最热最新枕边没有风浪怎么我们会跌荡yu 10 14pro等了快一个月才拿到拿到后感觉真香 2022年11月29日 14:34北京回复原味we 9 富士康那帮人还没解决好?加大产能啊 2022年11月29日 14:30广东东莞回复南玟芊芊 不止那一帮人 2022年11月29日 14:50河南新乡回复往鱼塘里扔炮仗 4 又说苹果今年销量下滑,没人买了,又说供不应求,我快要裂开了 2022年11月29日 14:28辽宁抚顺回复我去练琴了198701 1 你看最后一句,需求产量比是三比一,苹果的财报或者市场销量报告都创历史新高了,其实是华为和荣耀大跌 2022年11月29日 14:37上海回复烟花很暖 2 在墨西哥建厂 那真的好厉害 2022年12月17日 12:01四川成都回复 jivhong_ 2 14与13 有啥区别? 2022年11月29日 14:23广东惠州回复我去练琴了198701 2 芯片,屏幕,运存,扬声器,主摄,续航,卫星,第三代超瓷晶,第三代结构光模组,算是全面升级,所以销量逆势创新高 2022年11月29日 14:33上海回复这里是鱿鱼2 1",
            "title": "聚焦科技圈:今天有哪些大事?",
            "url": "http://finance.sina.com.cn/zt_d/subject-1660881614",
        },
    ]
    rrf_chunks = [
        {
            "chunk_id": 468308814313816318,
            "content": "## 儿童健康\n\n• 本设备及其配件可能包含一些小零件，请将设备及其配件放置在儿童接触不到的地方。儿童可能在无意之中损坏本设备及其配件，或吞下小零件导致窒息或其他危险。\n\n• 本设备并非玩具，儿童应在成人监护下使用设备。\n\n• 使用未经认可或不兼容的电源、充电器或电池，可能引发火灾、爆炸或其他危险。",
            "file_title": "HUAWEI MateStation S 12代酷睿版 用户指南-(PUC,Windows11_02,zh-cn)",
            "item_name": "HUAWEI MateStation S 12代酷睿版",
        },
        {
            "chunk_id": 468308814313816319,
            "content": "## 电池安全\n\n• 如果更换不正确的型号的电池会有起火或爆炸的危险。\n\n• 请勿将电池暴露在高温处或发热产品的周围，如日照、取暖器、微波炉、烤箱或热水器等。电池过热可能引起爆炸。\n\n• 请勿将电池放置在极低气压环境中，可能导致电池爆炸或泄漏可燃液体或气体。",
            "item_name": "HUAWEI MateStation S 12代酷睿版",
        },
        {
            "chunk_id": 468308814313816314,
            "content": "## 用户指南\n\n了解计算机\n    外观介绍 1\n    键盘 2\n    开启和关闭计算机 3\n    F10 一键恢复出厂 3\n    获取精彩功能 3\n安全信息\n个人信息和数据安全\n法律声明\n\n![主机外观及接口说明](http://localhost:9000/knowledge-base/HUAWEI_MateStation_S_12代酷睿版_用户指南-(PUC,Windows11_02,zh-cn)/621be1d7752348735664e224925f70d7452e3f65643b55ea2507031c3dac50e4.jpg)",
            "item_name": "HUAWEI MateStation S 12代酷睿版",
        },
        {
            "chunk_id": 468308814313816322,
            "content": "## 个人信息和数据安全\n\n在使用设备的一些功能和第三方应用时，可能会因为操作不正确或其他原因导致您的个人信息或数据泄露或丢失，建议按以下方式加强保护您的个人信息。\n\n• 请将设备放置于安全区域，防止未经授权人员使用您的设备。\n\n• 建议不要阅读来自陌生人的信息或邮件，以免设备遭受病毒感染。",
            "file_title": "HUAWEI MateStation S 12代酷睿版 用户指南-(PUC,Windows11_02,zh-cn)",
            "item_name": "HUAWEI MateStation S 12代酷睿版",
        },
        {
            "chunk_id": 468308814313816320,
            "content": "## 维护和保养\n\n• 不建议您自行升级部件或更换模块。如有相关服务需求，请联系华为客户服务中心。\n\n• 请保持设备及其配件干燥。请勿使用微波炉或吹风机等外部加热设备对其进行干燥处理。\n\n• 请勿在温度过高或过低区域放置设备及其配件，否则可能导致设备故障、着火或爆炸。",
            "item_name": "HUAWEI MateStation S 12代酷睿版",
        },
        {
            "chunk_id": 468308814313816317,
            "content": "## 听力保护\n\n• 当您使用耳机收听音乐或通话时，建议使用音乐或通话所需的最小音量，以免损伤听力。长时间接触高音量可能会导致永久性听力损伤。\n\n• 在加油站（维修站）或靠近易燃物品、化学制剂等任何易燃易爆区域，请勿使用本设备，并遵守所有图形或文字的指示。在燃油或化学制剂存放和运输区或易爆场所内或周围，设备可能引起爆炸或起火。\n\n• 请勿将设备及其配件与易燃液体、气体或易爆物品放在同一箱子中存放或运输。",
            "item_name": "HUAWEI MateStation S 12代酷睿版",
        },
    ]
    rewritten_query = ""
    __state: QueryGraphState = {
        "rrf_chunks": rrf_chunks,
        "web_search_docs": web_search_docs,
        "rewritten_query": "HUAWEI MateStation12电脑使用电源有哪些需要注意的安全点？",
    }

    reRankNode = ReRankNode()
    __state = reRankNode.process(__state)
    logger = logging.getLogger()
    logger.info(__state)
