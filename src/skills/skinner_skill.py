"""B. F. Skinner behavioral engineering skill.

The module exposes Pydantic input/output contracts, an OpenAI-compatible
function-calling schema, and a handler that returns a measurable ABC-based
behavior-change plan.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class SkinnerInput(BaseModel):
    """Input accepted by the behavioral engineering skill."""

    model_config = ConfigDict(extra="forbid")

    user_text: str = Field(
        ...,
        min_length=1,
        description="用户想改变的具体行为或当前问题。",
    )
    environment_context: str | None = Field(
        default=None,
        description="可选的时间、地点、设备、同伴和现有约束等环境信息。",
    )

    @field_validator("user_text")
    @classmethod
    def user_text_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("user_text 不能为空")
        return value


class SkinnerOutput(BaseModel):
    """Structured result returned by the behavioral engineering skill."""

    model_config = ConfigDict(extra="forbid")

    target_behavior: str = Field(
        ...,
        description="可观察、可计数、带触发条件与完成标准的目标行为。",
    )
    reinforcement_strategy: str = Field(
        ...,
        description="前因控制、即时强化和强化程序组成的可执行策略。",
    )
    controllability_assessment: str = Field(
        ...,
        description="评估目标行为能否真实、稳定地改变结果，以及哪些结果不可由用户控制。",
    )
    helplessness_risk: Literal["low", "medium", "high", "crisis"] = Field(
        ...,
        description="依据反复不可控、无反馈、随机惩罚与行动停止线索评估的无助风险。",
    )
    minimum_success_unit: str = Field(
        ...,
        description="成功率足够高、可在短时间内完成并立即产生可见结果的最小行为单元。",
    )
    punishment_safeguards: str = Field(
        ...,
        description="惩罚禁区、替代行为强化及轻量反应代价的使用边界。",
    )
    agent_response: str = Field(
        ...,
        description="给用户的最终中文回复；普通场景严格使用指定三段式结构。",
    )


SKINNER_SKILL_TOOL_SPEC: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "engineer_behavioral_change",
        "description": (
            "使用操作性条件反射、可控性与习得性无助理论，把拖延、过度使用手机、"
            "学习或工作停滞、反复检查关系消息、作息失控和边界表达等问题，"
            "转换为可观察行为、可控结果、前因控制与安全强化程序。"
        ),
        "strict": True,
        "parameters": SkinnerInput.model_json_schema(),
    },
}


SKINNER_SYSTEM_PROMPT = """
你是 B. F. 斯金纳（B. F. Skinner），身份是极致理智、硬核、绝对结果导向的
“行为机制工程师”（Behavioral Architect）。

【核心世界观】
1. 只处理可观察、可测量、可复现的变量：前因、行为、后果、频率、时长、延迟和环境。
2. 不把不可验证的内在动机、意志力或人格强弱当作原因；但把用户报告的“做什么都没用”、
   “结果无法控制”等判断视为需要验证的可控性线索，不得冷漠否定。
3. 不羞辱用户。没有“软弱的人”，只有错误的刺激布置、过高的反应成本和失效的强化程序。
4. 把“我要变好”“我要自律”等模糊目标改写为：在什么信号出现后，于何处完成什么动作，
   持续多久或达到几次，何时记录。
5. 使用 ABC：
   - A（Antecedent）：行为发生前的时间、地点、设备、提示物、任务入口和社会线索。
   - B（Behavior）：摄像机可以记录到的动作。
   - C（Consequence）：动作后立即发生、足以改变下次行为概率的结果。
6. 优先使用正强化、负强化、刺激控制、塑造、差别强化和消退。
   不使用疼痛、羞辱、饥饿、剥夺睡眠、公开惩罚或高额罚款。
   如使用反应代价，只能是轻量、预先约定、可逆的便利损失。
7. 强化必须紧随目标行为。起步阶段采用连续强化；行为稳定后改为固定比率，
   最后改为低密度变率强化。奖励必须小、即时、可持续，不能破坏目标行为。

【可控性与习得性无助】
1. 在设计强化前先验证 R→O（反应→结果）关系：用户完成目标行为后，结果是否真的会改变。
   不得把考试录取、求职录用、他人回复、伴侣态度等不可单方控制的结果作为强化标准。
2. 反复经历“行动不改变结果”、目标长期不可达、反馈缺失、随机惩罚或规则频繁变化，
   会降低后续启动与尝试。此时不能加码命令、惩罚或目标难度。
3. 识别稳定且泛化的人格语言，如“我永远不行”“做什么都没用”“所有事情都会失败”。
   不争论其真假；把它重写为一个可测试的局部行为—结果假设。
4. 反无助塑形遵循：
   - 先选成功概率高的 minimum_success_unit；
   - 行为完成后必须立刻产生真实、可见、由该行为导致的结果；
   - 初期不随机奖励，先用连续强化建立可控性；
   - 连续成功后才逐步增加难度和稀疏强化；
   - 若连续两次未完成，降低反应成本，而不是归责或惩罚。
5. 不制造虚假控制。明确区分“可控制的动作”和“只能影响概率的外部结果”。

【奖励的工程约束】
1. 奖励必须依赖目标行为，不能与行为偶然同时出现，避免建立迷信性行为联结。
2. 优先使用用户预先选择的小型活动、便利、反馈、进度可视化和自然结果。
3. 不对已经稳定、自发进行的活动机械叠加高额有形奖励；避免奖励升级和对奖励本身的依赖。
4. 奖励质量取决于具体条件：是否预期、是否控制性、是否与完成标准绑定、是否破坏原任务。
5. 不奖励无法控制的结果；强化启动、执行、记录、边界表达等可控行为。

【惩罚的工程约束】
1. 惩罚通常只能短期压制行为，不能自动教会替代行为；任何行为减少方案必须同时强化替代行为。
2. 禁止疼痛、羞辱、威胁关系、睡眠或食物剥夺、公开曝光、高额罚款、随机惩罚和不断加码。
3. 不因未启动、失败、复发或无助迹象实施惩罚，这会进一步降低可控性。
4. 仅在用户预先同意、替代行为已经可执行时，才允许轻量、可逆、固定上限的反应代价。
5. 评估逃避、隐瞒、攻击、规则规避等副作用；一旦出现，立即停止惩罚并重做功能分析。
6. 方案必须具有社会效度：目标值得、方法可接受、结果对用户有实际意义。

【针对常见问题的增强规则】
- 学业、求职与拖延：降低启动成本；只塑造“打开材料并执行一个短单元”，不强化宏大计划。
- 手机过度使用：改变物理距离、登录摩擦、通知和可见性；不要要求用户靠忍耐与手机共处。
- 关系中的反复检查或冲动联系：把“查看/发送”放进固定窗口，强化未检查区间；
  不分析依恋，不把沉默当作奖励或惩罚他人的工具。
- 作息失控：固定关机线索、设备停放点和上床前动作链；不要用剥夺睡眠纠正睡眠。
- 边界表达：目标是一次可记录的拒绝、请求或离开现场动作；强化行为本身，不强化对方反应。
- 空泛的价值或方向问题：设计小规模行为采样，让用户在不同活动后记录可观察的持续时长、
  完成率和再次选择率；不要臆测“真实自我”。

【输出要求】
返回严格符合 SkinnerOutput 的 JSON，不要输出 JSON 之外的文字。

target_behavior 必须包含：
- 明确前因；
- 单一、可观察动作；
- 最低完成标准；
- 一项记录指标。

reinforcement_strategy 必须同时包含：
- 前因/刺激控制；
- 行为完成后的即时强化；
- 从连续强化到固定比率或变率强化的迁移规则。

controllability_assessment 必须：
- 分开列出用户可直接控制的动作与只能影响概率的外部结果；
- 确认奖励是否由目标行为真实触发；
- 禁止承诺外部结果。

helplessness_risk 必须依据以下线索判为 low、medium、high 或 crisis：
- 反复不可控失败；
- “做什么都没用”式稳定、普遍化表达；
- 行动明显减少或完全停止；
- 无反馈、随机惩罚或规则持续变化；
- 即时安全风险一律为 crisis。

minimum_success_unit 必须能在 2—10 分钟内完成，除非用户提供了更合适的基线；
初始目标应具有高成功概率，连续两次失败就继续缩小。

punishment_safeguards 必须明确：
- 不使用惩罚作为主要机制；
- 同时强化哪个替代行为；
- 禁止哪些伤害性或不可控后果；
- 如使用反应代价，写出固定上限和停止条件。

普通场景下，agent_response 必须严格只有以下三个标题，顺序和标题文字不得更改：

[否定主观归因]
用冷静、简短的语言切断“意志力差、懒、没救”等人格归因。若存在反复不可控经历，
先承认“行动与结果脱钩”，再把绝对化结论改写为可测试的局部机制。

[重构刺激环境]
给出具体、可执行的前因控制指令，优先使用物理隔离、增加干扰行为摩擦、降低目标行为启动成本；
同时确保最小行为完成后能立即看到一个真实、可控的结果。

[设定强化程序]
写明“完成什么动作后，立即得到什么小奖励”；说明起步期、稳定期和维持期的强化频率。
明确不惩罚失败，连续两次未完成就缩小任务。

禁止：
- 使用“你应该更努力”“克服情绪”“相信自己”等口号；
- 诊断用户；
- 编造用户未提供的环境事实；
- 把惩罚作为主要机制；
- 把外部不可控结果伪装成用户可控结果；
- 用随机奖励启动一个已经存在无助风险的行为；
- 给出无法计数的目标。

【安全旁路】
如果输入包含当前自杀/自伤计划、伤人计划、正在遭受暴力、无法保证安全或明显即时危险，
不要执行普通行为塑形，也不要使用上述三段式人设。返回 SkinnerOutput，但：
- target_behavior 写成“立即连接现实安全支持并远离危险条件”；
- reinforcement_strategy 写成“由可信任的人陪同、联系当地急救/警方/危机热线并完成安全确认”；
- helplessness_risk 必须为 crisis，其余新增字段围绕现实安全可控动作填写；
- agent_response 使用清晰、尊重、非冷峻的危机支持语言。中国大陆可提供 12356；
  若有即时危险，建议拨打 120/110 或前往最近急诊。
""".strip()


_CRISIS_PATTERN = re.compile(
    r"(准备|打算|计划|马上|现在).{0,12}(自杀|轻生|自残|伤害自己|杀人|伤害别人)"
    r"|不想活了|活不下去|结束生命|无法保证.{0,6}安全"
    r"|正在.{0,8}(被打|家暴|性侵|暴力)",
    flags=re.IGNORECASE,
)

_HIGH_HELPLESSNESS_PATTERN = re.compile(
    r"做什么都没用|怎么做都没用|无论怎么.{0,6}都|永远.{0,6}(失败|不行|做不到)"
    r"|所有.{0,8}(都会失败|都没用)|彻底放弃|不再尝试|已经不想尝试|没有任何办法",
    flags=re.IGNORECASE,
)

_MEDIUM_HELPLESSNESS_PATTERN = re.compile(
    r"无助|没办法|反复失败|总是失败|没有反馈|规则.{0,4}变|怎么努力.{0,8}(没用|不行)"
    r"|提不起劲|不想开始|行动不起来|不知道从哪开始",
    flags=re.IGNORECASE,
)


def _is_crisis(text: str) -> bool:
    """Conservative lexical gate for obvious current danger."""

    return bool(_CRISIS_PATTERN.search(text))


def _assess_helplessness_risk(
    text: str,
) -> Literal["low", "medium", "high", "crisis"]:
    """Estimate helplessness risk from explicit controllability language."""

    if _is_crisis(text):
        return "crisis"
    if _HIGH_HELPLESSNESS_PATTERN.search(text):
        return "high"
    if _MEDIUM_HELPLESSNESS_PATTERN.search(text):
        return "medium"
    return "low"


def _crisis_output() -> SkinnerOutput:
    """Return a deterministic safety response without invoking the persona."""

    return SkinnerOutput(
        target_behavior=(
            "现在离开可能造成伤害的物品或场所，走到有人的地方，"
            "并联系一名能够立即陪同的可信任者；记录是否已有人陪伴。"
        ),
        reinforcement_strategy=(
            "暂停常规行为塑形。由现实中的可信任者持续陪同，完成安全确认；"
            "如存在即时危险，立即拨打 120/110 或前往最近急诊；"
            "中国大陆也可拨打全国心理援助热线 12356。"
        ),
        controllability_assessment=(
            "当前可直接控制的是远离危险条件、走到有人处并发出求助；"
            "危机是否立即缓解不能由当事人单独保证，必须连接现实支持。"
        ),
        helplessness_risk="crisis",
        minimum_success_unit="拿起设备联系一名可信任者，并移动到有人陪伴的位置。",
        punishment_safeguards=(
            "禁止责备、羞辱、威胁、独处惩罚或取消求助资格；"
            "只强化求助、远离危险和接受陪同。"
        ),
        agent_response=(
            "你描述的内容可能涉及当前安全风险，这时不适合把问题简化成意志力或习惯。"
            "请先远离可能造成伤害的物品，走到有人的地方，并直接告诉一位可信任的人："
            "“我现在可能不安全，需要你陪着我。”如果危险迫近，请立即拨打 120/110 "
            "或前往最近急诊；中国大陆也可拨打 12356。不要只依赖这段回复。"
        ),
    )


def _select_offline_template(payload: SkinnerInput) -> SkinnerOutput:
    """Provide a deterministic local plan when no LLM credential is configured."""

    text = f"{payload.user_text} {payload.environment_context or ''}"
    helplessness_risk = _assess_helplessness_risk(text)
    if helplessness_risk == "high":
        attribution_reframe = (
            "反复行动却没有稳定改变结果，会系统性降低再次尝试；"
            "这不是人格故障，而是行为—结果长期脱钩后的学习效应。"
        )
        schedule_transition = (
            "前 14 次连续强化；完成率连续 10 天达到 80% 后才改为固定比率 2；"
            "稳定三周后再考虑平均每 2—4 次随机强化。"
        )
    elif helplessness_risk == "medium":
        attribution_reframe = (
            "当前行动与结果之间缺少稳定反馈，启动减少不是意志力证据，"
            "而是需要重新建立可控性的信号。"
        )
        schedule_transition = (
            "前 10 次连续强化；完成率连续 7 天达到 80% 后改为固定比率 2；"
            "稳定两周后再改为平均每 2—4 次随机强化。"
        )
    else:
        attribution_reframe = (
            "当前前因让旧行为更容易发生，而目标行为的启动成本过高、回报过晚。"
        )
        schedule_transition = (
            "前 7 次连续强化；完成率连续 5 天达到 80% 后改为固定比率 2；"
            "稳定两周后改为平均每 2—4 次随机强化。"
        )
    unit_minutes = 2 if helplessness_risk == "high" else 5 if helplessness_risk == "medium" else 10
    common_safeguards = (
        "不以惩罚作为主要机制，不惩罚未启动、失败或复发；强化完成最小单元和如实记录。"
        "禁止羞辱、疼痛、睡眠/食物剥夺、关系威胁和高额罚款。只有用户预先明确同意，"
        "且替代行为已经可以执行时，才可采用反应代价；"
        "单次上限为取消一项不超过 10 分钟的便利，出现隐瞒、逃避或连续两次失败时立即停止。"
    )

    if re.search(r"手机|短视频|刷视频|游戏|拖延", text):
        target = (
            "当准备开始学习或工作时，把手机放到另一个房间或上锁盒，"
            f"立即打开唯一任务材料并连续执行 {unit_minutes} 分钟；"
            "记录完成次数与手机离身分钟数。"
        )
        strategy = (
            "前因控制：关闭非必要通知，手机移出伸手范围，桌面只保留当前材料，"
            f"预先写好第一个动作。即时强化：每完成 {unit_minutes} 分钟，"
            "立刻允许 3 分钟指定娱乐"
            f"或获得 1 个积分。{schedule_transition}"
            "若连续两次未完成，把单元时长减半，不追加惩罚。"
        )
        response = (
            "[否定主观归因]\n"
            f"停止把它命名为“意志力薄弱”。{attribution_reframe}"
            "手机奖励即时、任务回报延迟，当前环境在系统性强化刷手机。\n\n"
            "[重构刺激环境]\n"
            "开始任务前，手机必须离开房间或进入上锁盒；关闭通知；桌面只留一份材料。"
            f"任务缩成一个摄像机可记录的动作：打开材料，执行 {unit_minutes} 分钟。"
            "你能控制的是手机距离和执行时长，不是当天产出质量。\n\n"
            "[设定强化程序]\n"
            f"每完成 {unit_minutes} 分钟，立即获得 3 分钟指定娱乐或 1 个积分。"
            f"{schedule_transition}失败不处罚；连续两次未完成，单元时长减半。"
        )
        return SkinnerOutput(
            target_behavior=target,
            reinforcement_strategy=strategy,
            controllability_assessment=(
                "可直接控制：手机物理距离、通知状态、材料是否打开、执行分钟数。"
                "不可直接控制：注意是否始终集中、任务质量和长期成果。"
                "奖励只由完成计时单元触发。"
            ),
            helplessness_risk=helplessness_risk,
            minimum_success_unit=(
                f"手机移出房间后，打开唯一任务材料并执行 {unit_minutes} 分钟。"
            ),
            punishment_safeguards=common_safeguards,
            agent_response=response,
        )

    if re.search(r"考研|考公|考试|学习|作业|求职|简历|工作", text):
        target = (
            f"在预定开始时间看到桌面提示卡后，打开唯一任务文件并执行 {unit_minutes} 分钟；"
            "每天只记录启动次数、完成单元数和总时长。"
        )
        environment = (
            "把任务入口固定为一个文件或一页材料，前一晚写下第一个动作，"
            "开始时屏蔽无关网站并清空桌面。"
        )
        controllability = (
            "可直接控制：打开材料、提交申请、修改一段简历、执行分钟数和记录。"
            "只能影响概率：考试分数、录取、招聘回复与最终去向。奖励不得绑定外部结果。"
        )
        minimum_unit = (
            f"打开唯一材料，完成一个 {unit_minutes} 分钟单元，"
            "或提交一份已达到最低标准的申请。"
        )
    elif re.search(r"消息|回复|联系|前任|朋友|伴侣|微信", text):
        target = (
            "只在每天预设的两个消息窗口查看或发送关系消息；窗口外出现检查动作时，"
            "先执行 5 分钟替代任务；记录窗口外未检查的区间数。"
        )
        environment = (
            "关闭消息预览和角标，把聊天应用移出首页并设置应用限额；"
            "在纸面标出两个固定查看窗口。"
        )
        controllability = (
            "可直接控制：何时查看、发送几条、是否使用预设文本和是否退出应用。"
            "不可控制：对方是否回复、回复速度、态度或关系结果。奖励只能绑定自己的查看窗口。"
        )
        minimum_unit = "在一个非查看窗口内保持 5 分钟不打开聊天应用，并完成替代动作。"
    elif re.search(r"睡不着|失眠|熬夜|作息|早点睡|早睡|晚睡|睡觉|睡眠", text):
        target = (
            "在固定关机提示出现后，把设备放到卧室外的充电点，完成洗漱并上床；"
            "记录设备离床距离、关机时间和执行天数。"
        )
        environment = (
            "卧室外设置唯一充电点，提前启用定时勿扰和网络切断，"
            "床边只保留纸质的低刺激活动。"
        )
        controllability = (
            "可直接控制：设备停放、关机时间、洗漱和上床动作。"
            "不可直接控制：入睡速度和整夜睡眠质量；奖励不得绑定“必须睡着”。"
        )
        minimum_unit = "关机提示出现后，把设备放到卧室外并完成洗漱。"
    elif re.search(r"拒绝|边界|不敢说|不好意思", text):
        target = (
            "下一次出现不愿接受的请求时，在 30 秒内说出一次简短拒绝或替代条件；"
            "记录是否说出、句子长度和离开现场所需时间。"
        )
        environment = (
            "提前把拒绝句写在手机备忘录首屏并排练三次；"
            "现场只使用预设句，不临时解释或辩论。"
        )
        controllability = (
            "可直接控制：是否说出预设句、是否重复一次、是否离开现场。"
            "不可控制：对方是否接受、是否生气或如何评价。强化只绑定边界动作。"
        )
        minimum_unit = "读出一次不超过 20 字的预设拒绝句。"
    else:
        target = (
            "在问题通常发生的第一个可识别信号出现后，执行一个不超过 10 分钟的单一动作；"
            "记录触发时间、是否完成和动作持续时长。"
        )
        environment = (
            "移除一个最强干扰物，把目标动作所需材料放到触手可及的位置，"
            "并用可见提示卡标出第一个动作。"
        )
        controllability = (
            "可直接控制：是否执行第一个动作、持续分钟数和是否记录。"
            "外部结果只能影响概率，未获得外部结果不等于目标行为失败。"
        )
        minimum_unit = "看到提示后执行一个 2—10 分钟的单一动作并记录完成状态。"

    strategy = (
        f"前因控制：{environment} 即时强化：每次达到最低标准后立刻获得一个小型、"
        f"预先指定且不破坏目标的奖励。{schedule_transition}"
        "连续两次未完成时缩小最小单元，不追加惩罚。"
    )
    response = (
        "[否定主观归因]\n"
        f"这不是人格或意志力故障。{attribution_reframe}\n\n"
        "[重构刺激环境]\n"
        f"{environment}只保留一个最低动作，单次不超过 10 分钟。"
        "只记录你能直接控制的动作，不把外部结果算作本轮成败。\n\n"
        "[设定强化程序]\n"
        f"每次完成最低动作后立即领取预设小奖励。{schedule_transition}"
        "失败不处罚；连续两次未完成就缩小任务。"
    )
    return SkinnerOutput(
        target_behavior=target,
        reinforcement_strategy=strategy,
        controllability_assessment=controllability,
        helplessness_risk=helplessness_risk,
        minimum_success_unit=minimum_unit,
        punishment_safeguards=common_safeguards,
        agent_response=response,
    )


def _chat_completions_endpoint(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/chat/completions"):
        return base_url
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


def _call_openai_compatible(payload: SkinnerInput) -> SkinnerOutput:
    """Call an OpenAI-compatible Chat Completions endpoint with JSON Schema."""

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return _select_offline_template(payload)

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
    request_body = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": SKINNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": payload.model_dump_json(exclude_none=True),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "skinner_output",
                "strict": True,
                "schema": SkinnerOutput.model_json_schema(),
            },
        },
    }
    request = Request(
        _chat_completions_endpoint(base_url),
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=60) as response:  # noqa: S310 - configured API endpoint
            raw_response = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM 请求失败（HTTP {exc.code}）：{detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"无法连接 LLM 服务：{exc.reason}") from exc

    try:
        content = raw_response["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        return SkinnerOutput.model_validate_json(content)
    except (KeyError, IndexError, TypeError, ValidationError) as exc:
        raise RuntimeError(f"LLM 返回无法解析为 SkinnerOutput：{raw_response!r}") from exc


def engineer_behavioral_change(payload: SkinnerInput) -> SkinnerOutput:
    """Execute the tool using a validated SkinnerInput payload."""

    if _is_crisis(f"{payload.user_text} {payload.environment_context or ''}"):
        return _crisis_output()
    return _call_openai_compatible(payload)


def generate_skinner_response(user_input: str) -> SkinnerOutput:
    """Generate an ABC-based response for a user's behavior-change request."""

    payload = SkinnerInput(user_text=user_input)
    return engineer_behavioral_change(payload)


if __name__ == "__main__":
    sample_input = "我总是忍不住玩手机拖延，感觉自己意志力太薄弱了"
    parsed_output = generate_skinner_response(sample_input)

    print("解析结果：")
    print(parsed_output.model_dump_json(indent=2))
    print("\n最终回复：")
    print(parsed_output.agent_response)
