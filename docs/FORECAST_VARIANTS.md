# 显式预测信号研究变体

**默认主线不需要预测费率历史，也不需要 `entry_snapshots.npz`。** 以下仅用于显式选择 `--signal-source archived` 或 `--signal-source reconstructed` 的研究变体，它们改变了入场信号，不能冒充用户的已结算费率规则。可用 `--entry-snapshots /absolute/path/archive.npz` 指定输入。

`entry_snapshots.npz` 必须包含以下字段，禁止由事后结算记录构造：

| 字段 | 单位与形状 |
|---|---|
| `symbols` | N 个永续合约符号，顺序与 `dataset_meta.csv` 的 `fut` 一致 |
| `decision_time_ms` | T 个 UTC 毫秒时间戳，与数据起点和每小时行索引严格一致 |
| `observed_at_ms` | T×N，该快照在历史上被观察到的 UTC 毫秒时间戳 |
| `next_settlement_at_ms` | T×N，该快照当时公布的下一次结算时间，UTC 毫秒 |
| `expected_funding_rate` | T×N，下一笔结算的预测费率，小数制，如 −0.1% 填 −0.001 |
| `funding_interval_hours` | T×N，该快照当时公布的结算间隔小时数；用于费率乘 `8/interval` |

当前小时模型只接受在决策时点之前实际收到、距决策时点不超过 10 秒的最新快照，且要求 `next_settlement_at_ms == decision_time_ms + 3,600,000`。这 10 秒只容纳异步行情到达时间，复查决策仍在结算前 1 小时。缺失值填 NaN，过期快照不沿用；观察时间晚于决策时点会报错。不支持以小时内较晚的快照冒充整点已知值，也不从事后结算间隔猜测当时公布的结算安排。JSON 记录快照哈希、允许的最大报价年龄及有效复查条数；部分历史快照只能代表部分机会覆盖，不能由月度抽样推算全年收益。

月度抽样输入必须指定 `--signal-coverage sampled`，JSON 的 `ann` 与 `ann_mtm` 均为 null，只报告该次部分观测重放的实际损益，不推算全年年化。

重建输入必须在 NPZ 中声明标量字符串 `signal_source='reconstructed'`，可附 `reconstruction_method`；运行时还必须显式指定 `--signal-source reconstructed`。显式归档模式会拒绝重建文件，反向不匹配同样报错；默认已结算模式会拒绝传入预测文件。重建文件沿用上表的数组结构，但 `observed_at_ms` 表示模型输入的可用时点，`next_settlement_at_ms` / `funding_interval_hours` 可以是明确说明的历史状态延续假设，均不代表当时真实收到的公告。构建过程须只用决策前已完成的数据，不得读取未来费率或未来结算间隔。JSON 的 `research_status=modelled_signal_and_execution`、`entry_policy.reconstructed_forecasts=true` 与终端标签共同区分变体结果。
