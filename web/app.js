const statusNode = document.querySelector('#status');
const runButton = document.querySelector('#run');
const symbolInput = document.querySelector('#symbol');
const endDateInput = document.querySelector('#end-date');
const datalist = document.querySelector('#symbols');
const showSignals = document.querySelector('#show-signals');
const showNumbers = document.querySelector('#show-numbers');
const chartNode = document.querySelector('#chart');

let pyodide;
let manifest;
let chart;
let resizeObserver;
let currentResult;

function setStatus(message, error = false) {
  statusNode.textContent = message;
  statusNode.classList.toggle('error', error);
}

function shardFor(symbol) {
  let crc = 0xffffffff;
  for (const byte of new TextEncoder().encode(symbol)) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) {
      crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
    }
  }
  return ((crc ^ 0xffffffff) >>> 0) % manifest.shards;
}

async function loadSymbol(symbol) {
  const shard = shardFor(symbol).toString(16).padStart(2, '0');
  const response = await fetch(`./data/shards/${shard}.json.gz`);
  if (!response.ok) throw new Error(`数据分片加载失败: HTTP ${response.status}`);
  if (!globalThis.DecompressionStream) throw new Error('当前浏览器不支持 gzip 解压，请使用新版 Chrome、Edge、Firefox 或 Safari');
  const stream = response.body.pipeThrough(new DecompressionStream('gzip'));
  const payload = JSON.parse(await new Response(stream).text());
  if (!payload[symbol]) throw new Error(`没有找到 ${symbol}`);
  return payload[symbol];
}

function trimPayload(payload, endDate) {
  const start = new Date(`${payload.s}T00:00:00Z`);
  const end = new Date(`${endDate}T00:00:00Z`);
  let count = 0;
  while (count < payload.d.length) {
    const current = new Date(start);
    current.setUTCDate(current.getUTCDate() + payload.d[count]);
    if (current > end) break;
    count += 1;
  }
  return Object.fromEntries(Object.entries(payload).map(([key, value]) => [
    key,
    Array.isArray(value) ? value.slice(0, count) : value,
  ]));
}

function visibleResult(result) {
  const visible = items => items.filter(item => item.time >= manifest.cutoff);
  return {
    bars: visible(result.bars),
    markers: visible(result.markers),
    numbers: visible(result.numbers),
  };
}

function render() {
  if (!currentResult) return;
  if (resizeObserver) resizeObserver.disconnect();
  if (chart) chart.remove();
  chart = LightweightCharts.createChart(chartNode, {
    width: chartNode.clientWidth,
    height: chartNode.clientHeight,
    layout: { background: { color: '#ffffff' }, textColor: '#34404c' },
    grid: { vertLines: { color: '#eef1f4' }, horzLines: { color: '#eef1f4' } },
    rightPriceScale: { borderColor: '#dce2e8' },
    timeScale: { borderColor: '#dce2e8', timeVisible: true },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
  });
  const candles = chart.addCandlestickSeries({
    upColor: '#00897b', downColor: '#d84343', borderVisible: false,
    wickUpColor: '#00897b', wickDownColor: '#d84343',
  });
  candles.setData(currentResult.bars);
  const markers = [];
  if (showSignals.checked) markers.push(...currentResult.markers);
  if (showNumbers.checked) markers.push(...currentResult.numbers);
  markers.sort((a, b) => a.time.localeCompare(b.time));
  candles.setMarkers(markers);
  chart.timeScale().fitContent();
  resizeObserver = new ResizeObserver(([entry]) => chart.applyOptions({
    width: entry.contentRect.width,
    height: entry.contentRect.height,
  }));
  resizeObserver.observe(chartNode);
}

async function calculate() {
  const symbol = symbolInput.value.trim().toUpperCase();
  if (!manifest.symbols.includes(symbol)) return setStatus(`没有找到 ${symbol}`, true);
  runButton.disabled = true;
  setStatus(`正在加载 ${symbol}...`);
  try {
    const payload = trimPayload(await loadSymbol(symbol), endDateInput.value);
    if (!payload.d.length) throw new Error('截止日期早于可用行情');
    pyodide.globals.set('payload_json', JSON.stringify(payload));
    const output = await pyodide.runPythonAsync('from s1demo import calculate_json\ncalculate_json(payload_json)');
    currentResult = visibleResult(JSON.parse(output));
    render();
    const actualEnd = currentResult.bars.at(-1).time;
    setStatus(`${symbol} · ${currentResult.bars.length} 根 · ${manifest.cutoff} 至 ${actualEnd}`);
  } catch (error) {
    setStatus(error.message || String(error), true);
  } finally {
    runButton.disabled = false;
  }
}

async function initialize() {
  const response = await fetch('./data/manifest.json');
  if (!response.ok) throw new Error(`清单加载失败: HTTP ${response.status}`);
  manifest = await response.json();
  endDateInput.min = manifest.cutoff;
  endDateInput.max = manifest.latest;
  endDateInput.value = manifest.latest;
  datalist.replaceChildren(...manifest.symbols.map(symbol => Object.assign(document.createElement('option'), { value: symbol })));
  pyodide = await loadPyodide();
  await pyodide.loadPackage(['numpy', 'pandas']);
  const files = ['__init__.py', 'runtime.py', 'signals.py', 'indicators.py', 'auxiliary_tree_model.py', 'causal_auxiliary_features.py', 'causal_auxiliary_tree_model.py', 'auxiliary_signals.py', 'monotonic_zigzag.py', 'zigzag_signals.py'];
  pyodide.FS.mkdirTree('/home/pyodide/s1demo');
  await Promise.all(files.map(async name => {
    const source = await fetch(`./python/s1demo/${name}`).then(fileResponse => {
      if (!fileResponse.ok) throw new Error(`${name} 加载失败`);
      return fileResponse.text();
    });
    pyodide.FS.writeFile(`/home/pyodide/s1demo/${name}`, source);
  }));
  pyodide.runPython("import sys\nsys.path.insert(0, '/home/pyodide')");
  runButton.disabled = false;
  setStatus(`${manifest.symbols.length.toLocaleString()} 只股票已就绪`);
  await calculate();
}

runButton.addEventListener('click', calculate);
symbolInput.addEventListener('keydown', event => { if (event.key === 'Enter') calculate(); });
endDateInput.addEventListener('change', calculate);
showSignals.addEventListener('change', render);
showNumbers.addEventListener('change', render);
initialize().catch(error => setStatus(error.message || String(error), true));
