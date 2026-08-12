# S1 Formula Demo

这是一个可直接部署到 GitHub Pages 的独立演示项目。页面在浏览器中运行 Python 公式，读取内置的不复权日线行情，绘制 K 线、B/S、IB/E 和连续数字。项目不需要后端服务，也不包含训练脚本、训练集或完整生产数据库。

## 在线演示结构

- `index.html` / `web/app.js`：静态页面和图表交互。
- `python/s1demo/`：信号计算代码。Pyodide 在浏览器中加载这些 Python 模块。
- `data/manifest.json`：股票列表、日期范围和分片元数据。
- `data/shards/*.json.gz`：256 个 gzip 行情分片。
- `tools/build_demo_data.py`：从本地生产 SQLite 重新生成演示数据。
- `tools/verify_demo.py`：检查清单、分片映射、数据列和公式运行结果。

当前数据包含 13,942 只股票，页面展示最近 251 个交易日，即 `2025-08-11` 至 `2026-08-10`。每只股票还携带最多 100 根隐藏暖机 K 线，用来初始化均线、成交额指标和 ZigZag 状态。暖机区间参与计算，但不会显示在图上。

## 为什么不把 SQLite 放到 GitHub

本地完整 SQLite 约 1.88 GiB，只保留一年仍预计超过 200 MiB，超过 GitHub 普通 Git 单文件 100 MB 限制。SQLite 还需要静态网页先下载整个数据库，浏览器才能查询其中一只股票，加载成本不合理。

演示数据改为“列数组 + gzip + 哈希分片”：

1. 日期保存为首日加整数偏移量。
2. OHLC、成交量、成交额和换手率按列保存，避免每行重复字段名。
3. `CRC32(symbol) % 256` 决定股票所在分片。
4. 浏览器只下载目标股票所在的一个分片，当前最大分片小于 400 KiB。

全部压缩行情约 61.7 MiB，单文件远低于 GitHub 限制。相对于 Parquet，这种格式不需要在浏览器额外加载 Arrow/Parquet 解析器；相对于每只股票一个文件，它也避免了仓库中出现约 14,000 个小文件。

生产项目仍然使用完整 SQLite。这个变化只针对公开演示仓库，不会替换本地数据结构。

## 信号计算

浏览器加载所选股票的行情后，将数据传给 `s1demo.calculate_json`：

- B/S 使用单向价格反转状态机：候选只能在当前 K 线出生，后续可以永久撤回，但不会首次回填到历史 K 线，也不会在撤回后恢复。
- S 出现后必须等待新的 B 才能再次出现 S；B 的过滤会保留低波动局部低点反弹和放量暴跌后的强势阳线反弹。
- IB/E 使用当前 `exact` 公式配置。
- 连续数字按方向状态和中断规则计算。
- 截止日期会先截断输入行情再计算，因此可以观察历史时点的可见结果。

IB/E 公式允许最多下一根 K 线参与确认，所以回放到历史截止日期时，最后一根附近的 IB/E 可能与事后完整数据不同。B/S 只允许从已有位置撤回，不允许未来回填或恢复。

## 本地运行

不能直接双击 `index.html`，浏览器会阻止静态文件之间的 `fetch`。在项目根目录启动任意静态服务器：

```powershell
python -m http.server 8780
```

然后打开 <http://localhost:8780/>。首次打开需要从 CDN 下载 Pyodide、NumPy、Pandas 和 Lightweight Charts，之后选择股票只加载一个本地数据分片。

校验仓库内置数据和公式：

```powershell
python tools/verify_demo.py
```

本地 Python 校验需要 `numpy` 和 `pandas`。网页本身不要求访问者安装 Python。

## 更新数据

数据生成器默认读取相邻主项目中的 `data/bfq_daily.sqlite3`：

```powershell
python tools/build_demo_data.py
```

也可以显式指定来源和区间：

```powershell
python tools/build_demo_data.py --source D:\path\bfq_daily.sqlite3 --visible-days 251 --warmup-days 100
```

生成器只读 SQLite，并重建 `data/shards` 和 `data/manifest.json`。更新完成后应再次运行校验并提交生成结果。

## GitHub Pages

1. 将本目录推送到 GitHub 仓库的 `main` 分支。
2. 在仓库 `Settings` -> `Pages` 中选择 `Deploy from a branch`。
3. 分支选择 `main`，目录选择 `/ (root)`。
4. 保存并等待 Pages 完成部署。

页面依赖 jsDelivr 上的 Pyodide 和 Lightweight Charts，因此访问端需要联网并使用支持 `DecompressionStream` 的现代浏览器。

## 数据说明

行情是本地导入的不复权日线数据，字段包括开盘、最高、最低、收盘、成交量、成交额和换手率。数据及信号仅用于技术演示，不构成投资建议。
