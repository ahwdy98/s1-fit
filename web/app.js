const statusNode = document.querySelector('#status');
const runButton = document.querySelector('#run');
const loadLatestButton = document.querySelector('#load-latest');
const symbolInput = document.querySelector('#symbol');
const startDateInput = document.querySelector('#start-date');
const endDateInput = document.querySelector('#end-date');
const datalist = document.querySelector('#symbols');
const showSignals = document.querySelector('#show-signals');
const showRemoved = document.querySelector('#show-removed');
const showNumbers = document.querySelector('#show-numbers');
const chartNode = document.querySelector('#chart');

let pyodide;
let manifest;
let chart;
let resizeObserver;
let currentResult;
let currentFullResult;
let currentCalculationKey;
const onlinePayloads = new Map();
let jsonpSequence = 0;

const EASTMONEY_MARKETS = ['105', '106', '107'];
const EASTMONEY_URL = 'https://push2his.eastmoney.com/api/qt/stock/kline/get';
const SINA_URL = 'https://stock.finance.sina.com.cn/usstock/api/jsonp.php';
const MARKET_CACHE_PREFIX = 's1demo:eastmoney-market:';

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

function requestJsonp(request, timeoutMs = 12000) {
  return new Promise((resolve, reject) => {
    const callbackName = `__s1demo_jsonp_${Date.now()}_${jsonpSequence += 1}`;
    const usesCallbackPath = typeof request === 'function';
    const url = usesCallbackPath ? request(callbackName) : request;
    const script = document.createElement('script');
    const timer = window.setTimeout(() => {
      cleanup();
      reject(new Error('在线行情请求超时'));
    }, timeoutMs);

    function cleanup() {
      window.clearTimeout(timer);
      script.remove();
      delete window[callbackName];
    }

    window[callbackName] = payload => {
      cleanup();
      resolve(payload);
    };
    script.onerror = () => {
      cleanup();
      reject(new Error('在线行情接口连接失败'));
    };
    if (!usesCallbackPath) url.searchParams.set('cb', callbackName);
    url.searchParams.set('_', String(Date.now()));
    script.src = url.toString();
    script.referrerPolicy = 'no-referrer-when-downgrade';
    script.charset = 'gbk';
    document.head.append(script);
  });
}

function normalizeProviderSymbol(symbol) {
  return symbol.replaceAll('_', '.');
}

function parseEastmoneyPayload(response, symbol, market) {
  const data = response?.data;
  const klines = data?.klines;
  if (!Array.isArray(klines) || klines.length < 20) return null;

  const rows = klines.map(value => String(value).split(','));
  const parsed = rows.map(columns => ({
    date: columns[0],
    open: Number(columns[1]),
    close: Number(columns[2]),
    high: Number(columns[3]),
    low: Number(columns[4]),
    volume: Number(columns[5]),
    amount: Number(columns[6]),
    turnoverRate: Number(columns[10]),
  })).filter(row => (
    /^\d{4}-\d{2}-\d{2}$/.test(row.date)
    && [row.open, row.close, row.high, row.low, row.volume].every(Number.isFinite)
  )).sort((left, right) => left.date.localeCompare(right.date));
  if (parsed.length < 20) return null;

  const firstDate = new Date(`${parsed[0].date}T00:00:00Z`);
  const dayOffsets = parsed.map(row => (
    Math.round((new Date(`${row.date}T00:00:00Z`) - firstDate) / 86400000)
  ));
  return {
    payload: {
      s: parsed[0].date,
      d: dayOffsets,
      o: parsed.map(row => row.open),
      h: parsed.map(row => row.high),
      l: parsed.map(row => row.low),
      c: parsed.map(row => row.close),
      v: parsed.map(row => Math.max(0, Math.round(row.volume))),
      a: parsed.map(row => (Number.isFinite(row.amount) ? row.amount : 0)),
      t: parsed.map(row => (Number.isFinite(row.turnoverRate) ? row.turnoverRate : 0)),
    },
    market,
    name: data.name || symbol,
    latest: parsed.at(-1).date,
    source: 'eastmoney',
    sourceLabel: '东方财富在线行情',
    missingTurnoverRows: 0,
  };
}

function parseSinaPayload(rows, symbol) {
  if (!Array.isArray(rows) || rows.length < 20) return null;
  const parsed = rows.map(row => ({
    date: String(row.d || ''),
    open: Number(row.o),
    close: Number(row.c),
    high: Number(row.h),
    low: Number(row.l),
    volume: Number(row.v),
    amount: Number(row.a),
  })).filter(row => (
    /^\d{4}-\d{2}-\d{2}$/.test(row.date)
    && [row.open, row.close, row.high, row.low, row.volume].every(Number.isFinite)
  )).sort((left, right) => left.date.localeCompare(right.date)).slice(-351);
  if (parsed.length < 20) return null;

  const firstDate = new Date(`${parsed[0].date}T00:00:00Z`);
  return {
    payload: {
      s: parsed[0].date,
      d: parsed.map(row => (
        Math.round((new Date(`${row.date}T00:00:00Z`) - firstDate) / 86400000)
      )),
      o: parsed.map(row => row.open),
      h: parsed.map(row => row.high),
      l: parsed.map(row => row.low),
      c: parsed.map(row => row.close),
      v: parsed.map(row => Math.max(0, Math.round(row.volume))),
      a: parsed.map(row => (Number.isFinite(row.amount) ? row.amount : 0)),
      t: parsed.map(() => 0),
    },
    market: '',
    name: symbol,
    latest: parsed.at(-1).date,
    source: 'sina',
    sourceLabel: '新浪在线行情',
    missingTurnoverRows: parsed.length,
  };
}

async function fetchEastmoneyMarket(symbol, market) {
  const url = new URL(EASTMONEY_URL);
  url.searchParams.set('secid', `${market}.${normalizeProviderSymbol(symbol)}`);
  url.searchParams.set('klt', '101');
  url.searchParams.set('fqt', '0');
  url.searchParams.set('lmt', '351');
  url.searchParams.set('end', '20500101');
  url.searchParams.set('fields1', 'f1,f2,f3,f4,f5,f6');
  url.searchParams.set('fields2', 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61');
  const response = await requestJsonp(url);
  return parseEastmoneyPayload(response, symbol, market);
}

async function fetchSinaSymbol(symbol) {
  const response = await requestJsonp(callbackName => (
    new URL(`${SINA_URL}/${callbackName}/US_MinKService.getDailyK?symbol=${encodeURIComponent(normalizeProviderSymbol(symbol))}`)
  ));
  const result = parseSinaPayload(response, symbol);
  if (!result) throw new Error(`新浪在线行情没有找到 ${symbol}`);
  return result;
}

function mergeEmbeddedTurnover(result, embeddedPayload) {
  const embeddedDates = payloadDates(embeddedPayload);
  const embeddedTurnover = new Map(
    embeddedDates.map((date, index) => [date, Number(embeddedPayload.t[index])]),
  );
  const onlineDates = payloadDates(result.payload);
  let missing = 0;
  result.payload.t = onlineDates.map((date, index) => {
    const value = embeddedTurnover.get(date);
    if (Number.isFinite(value)) return value;
    missing += 1;
    return Number(result.payload.t[index]) || 0;
  });
  return { ...result, missingTurnoverRows: missing };
}

async function loadLatestSymbol(symbol) {
  let cachedMarket = null;
  try {
    cachedMarket = localStorage.getItem(`${MARKET_CACHE_PREFIX}${symbol}`);
  } catch (_error) {
    // Private browsing can disable persistent storage; market probing still works.
  }
  const markets = [cachedMarket, ...EASTMONEY_MARKETS]
    .filter((value, index, values) => value && values.indexOf(value) === index);
  let lastError = null;
  for (const market of markets) {
    try {
      const result = await fetchEastmoneyMarket(symbol, market);
      if (!result) continue;
      try {
        localStorage.setItem(`${MARKET_CACHE_PREFIX}${symbol}`, market);
      } catch (_error) {
        // The in-memory result remains usable when localStorage is unavailable.
      }
      return result;
    } catch (error) {
      lastError = error;
      break;
    }
  }
  try {
    return await fetchSinaSymbol(symbol);
  } catch (error) {
    throw lastError || error || new Error(`在线行情没有找到 ${symbol}`);
  }
}

function payloadDates(payload) {
  const start = new Date(`${payload.s}T00:00:00Z`);
  return payload.d.map(offset => {
    const current = new Date(start);
    current.setUTCDate(current.getUTCDate() + offset);
    return current.toISOString().slice(0, 10);
  });
}

function useOnlineDateRange(payload) {
  const dates = payloadDates(payload);
  const latest = dates.at(-1);
  const visibleDays = manifest.visible_trading_days || 251;
  const defaultStart = dates[Math.max(0, dates.length - visibleDays)];
  startDateInput.min = dates[0];
  startDateInput.max = latest;
  startDateInput.value = defaultStart;
  endDateInput.min = dates[0];
  endDateInput.max = latest;
  endDateInput.value = latest;
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
  const startDate = startDateInput.value || manifest.cutoff;
  const visible = items => items.filter(item => item.time >= startDate);
  return {
    bars: visible(result.bars),
    markers: visible(result.markers),
    removedMarkers: visible(result.removedMarkers || []),
    numbers: visible(result.numbers),
    formulaProfile: result.formulaProfile,
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
  if (showRemoved.checked) markers.push(...currentResult.removedMarkers);
  if (showNumbers.checked) markers.push(...currentResult.numbers);
  markers.sort((a, b) => a.time.localeCompare(b.time) || a.text.localeCompare(b.text));
  candles.setMarkers(markers);
  chart.timeScale().fitContent();
  resizeObserver = new ResizeObserver(([entry]) => {
    chart.applyOptions({
      width: entry.contentRect.width,
      height: entry.contentRect.height,
    });
    chart.timeScale().fitContent();
  });
  resizeObserver.observe(chartNode);
}

async function calculate() {
  const symbol = symbolInput.value.trim().toUpperCase();
  const onlineSource = onlinePayloads.get(symbol);
  if (!onlineSource && !manifest.symbols.includes(symbol)) {
    return setStatus(`内置数据没有 ${symbol}，可尝试读取最新行情`, true);
  }
  if (startDateInput.value > endDateInput.value) return setStatus('开始日期不能晚于截止日期', true);
  runButton.disabled = true;
  loadLatestButton.disabled = true;
  const sourceKey = onlineSource
    ? `online:${onlineSource.source}:${onlineSource.market}:${onlineSource.latest}`
    : 'embedded';
  const calculationKey = `${symbol}:${endDateInput.value}:${sourceKey}`;
  setStatus(`正在计算 ${symbol}...`);
  try {
    if (currentCalculationKey !== calculationKey || !currentFullResult) {
      const sourcePayload = onlineSource?.payload || await loadSymbol(symbol);
      const payload = trimPayload(sourcePayload, endDateInput.value);
      if (!payload.d.length) throw new Error('截止日期早于可用行情');
      pyodide.globals.set('payload_json', JSON.stringify(payload));
      const output = await pyodide.runPythonAsync('from s1demo import calculate_json\ncalculate_json(payload_json)');
      currentFullResult = JSON.parse(output);
      currentCalculationKey = calculationKey;
    }
    currentResult = visibleResult(currentFullResult);
    if (!currentResult.bars.length) throw new Error('所选日期范围内没有行情');
    render();
    const actualStart = currentResult.bars[0].time;
    const actualEnd = currentResult.bars.at(-1).time;
    const sourceLabel = onlineSource?.sourceLabel || '内置行情';
    const warning = onlineSource?.missingTurnoverRows
      ? ` · ${onlineSource.missingTurnoverRows} 根缺少换手率，IB/E 为近似结果`
      : '';
    setStatus(`${symbol} · ${currentResult.bars.length} 根 · ${actualStart} 至 ${actualEnd} · ${sourceLabel}${warning}`);
  } catch (error) {
    setStatus(error.message || String(error), true);
  } finally {
    runButton.disabled = false;
    loadLatestButton.disabled = false;
  }
}

async function fetchLatestAndCalculate() {
  const symbol = symbolInput.value.trim().toUpperCase();
  if (!symbol) return setStatus('请输入股票代码', true);
  runButton.disabled = true;
  loadLatestButton.disabled = true;
  setStatus(`正在读取 ${symbol} 最新行情...`);
  try {
    let result = await loadLatestSymbol(symbol);
    if (result.source === 'sina' && manifest.symbols.includes(symbol)) {
      try {
        result = mergeEmbeddedTurnover(result, await loadSymbol(symbol));
      } catch (_error) {
        // Online OHLCV remains usable when the embedded enrichment fails.
      }
    }
    onlinePayloads.set(symbol, result);
    useOnlineDateRange(result.payload);
    currentFullResult = null;
    currentCalculationKey = null;
    await calculate();
  } catch (error) {
    setStatus(`${error.message || String(error)}，仍可使用内置行情`, true);
  } finally {
    runButton.disabled = false;
    loadLatestButton.disabled = false;
  }
}

async function initialize() {
  const response = await fetch('./data/manifest.json');
  if (!response.ok) throw new Error(`清单加载失败: HTTP ${response.status}`);
  manifest = await response.json();
  startDateInput.min = manifest.cutoff;
  startDateInput.max = manifest.latest;
  startDateInput.value = manifest.cutoff;
  endDateInput.min = manifest.cutoff;
  endDateInput.max = manifest.latest;
  endDateInput.value = manifest.latest;
  datalist.replaceChildren(...manifest.symbols.map(symbol => Object.assign(document.createElement('option'), { value: symbol })));
  pyodide = await loadPyodide();
  await pyodide.loadPackage(['numpy', 'pandas']);
  const files = ['__init__.py', 'runtime.py', 'signals.py', 'indicators.py', 'auxiliary_tree_model.py', 'causal_auxiliary_features.py', 'causal_auxiliary_tree_model.py', 'auxiliary_signals.py', 'futu_metrics.py', 'restricted_s1_formula.py', 'monotonic_zigzag.py', 'zigzag_signals.py'];
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
  loadLatestButton.disabled = false;
  setStatus(`${manifest.symbols.length.toLocaleString()} 只股票已就绪`);
  await calculate();
}

runButton.addEventListener('click', calculate);
loadLatestButton.addEventListener('click', fetchLatestAndCalculate);
symbolInput.addEventListener('keydown', event => { if (event.key === 'Enter') calculate(); });
endDateInput.addEventListener('change', calculate);
startDateInput.addEventListener('change', calculate);
showSignals.addEventListener('change', render);
showRemoved.addEventListener('change', render);
showNumbers.addEventListener('change', render);
initialize().catch(error => setStatus(error.message || String(error), true));
