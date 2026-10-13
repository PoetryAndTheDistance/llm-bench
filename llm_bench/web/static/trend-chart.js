/*
 * 任务趋势图公共组件（Locust 风格：墙钟时间轴 / 网格刻度 / 面积折线 / 十字线 tooltip）
 *
 * 实时压测页（dashboard.html）与离线页面快照（job-report-template.html 生成的
 * report.html）共用本文件，保证两处趋势图视觉与交互一致，避免两份实现各自漂移。
 * 快照要求单文件离线可打开，由 snapshot.py 在生成 report.html 时把本文件整体内联。
 *
 * DOM 约定（与 dashboard 一致）：
 *   图表面板（class="trend-surface"）：延迟 #ttftChart / #e2eChart；
 *     吞吐 #rpmChart / #decodeChart / #prefillChart / #effChart /
 *     #inputTpmChart / #outputTpmChart（存在即初始化，可按需只放部分）
 *   图例      #latencyLegend / #throughputLegend
 *
 * 用法：
 *   const charts = TrendCharts.create();
 *   charts.update(keys, history);  // keys: 当前分组名数组；history: [{t, data:{key: group}}]
 *   charts.reset();                // 新任务：清空颜色槽位注册
 *   charts.redraw();               // 容器从隐藏切回可见时重绘
 */
(function () {
  'use strict';

  const CSS = `
.trend-legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 4px 0 4px; min-height: 20px; }
.trend-legend .lg-item {
  display: inline-flex; align-items: center; gap: 7px;
  font-size: 12px; color: var(--text, #1e293b); cursor: pointer; user-select: none;
}
.trend-legend .lg-item.off { opacity: .35; }
.trend-legend .lg-dot { width: 16px; height: 3px; border-radius: 2px; flex-shrink: 0; }
.trend-legend .lg-empty { font-size: 12px; color: var(--text-light, #64748b); }
.trend-panel {
  border: 1px solid var(--border, #e2e8f0); border-radius: 8px;
  padding: 12px 14px 8px; margin-top: 14px; background: #fff;
}
.trend-title { font-size: 13px; font-weight: 600; color: var(--text, #1e293b); }
.trend-title .unit { font-weight: 400; color: var(--text-light, #64748b); margin-left: 6px; }
.trend-surface { position: relative; height: 220px; margin-top: 2px; }
.trend-surface svg { display: block; width: 100%; height: 100%; }
.trend-empty { fill: #94a3b8; font-size: 12px; }
.trend-tip {
  position: absolute; pointer-events: none; z-index: 5; display: none;
  background: rgba(255,255,255,.98); border: 1px solid var(--border, #e2e8f0);
  border-radius: 6px; box-shadow: 0 4px 14px rgba(15,23,42,.12);
  padding: 7px 10px; font-size: 11px; line-height: 1.7; white-space: nowrap;
}
.trend-tip .tip-time { color: var(--text-light, #64748b); margin-bottom: 2px; font-variant-numeric: tabular-nums; }
.trend-tip .tip-row { display: flex; align-items: center; gap: 7px; color: var(--text, #1e293b); }
.trend-tip .tip-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.trend-tip .tip-val { margin-left: 12px; font-variant-numeric: tabular-nums; }
`;

  function injectCss() {
    if (document.querySelector('style[data-trend-chart]')) return;
    const style = document.createElement('style');
    style.dataset.trendChart = '1';
    style.textContent = CSS;
    document.head.appendChild(style);
  }

  function escHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, c => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
  }

  const SVG_NS = 'http://www.w3.org/2000/svg';
  // 趋势图多供应商/模型分组配色：固定槽位顺序（已通过色觉校验），颜色跟随实体，不随排名变化
  const SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948'];
  // 超过 8 个分组：灰色 + 不同虚线作为次级编码
  const EXTRA_DASHES = ['', '4 3', '1 3', '7 3 1 3'];

  const CHART_META = {
    ttft: { group: 'latency', unit: 'ms', pick: g => g.ttft && g.ttft.p50 != null ? g.ttft.p50 * 1000 : null },
    e2e: { group: 'latency', unit: 'ms', pick: g => g.e2e && g.e2e.p50 != null ? g.e2e.p50 * 1000 : null },
    rpm: { group: 'throughput', unit: 'rpm', pick: g => g.rpm != null ? g.rpm : null },
    decode: { group: 'throughput', unit: 't/s', pick: g => g.decode_tps && g.decode_tps.p50 != null ? g.decode_tps.p50 : null },
    prefill: { group: 'throughput', unit: 't/s', pick: g => g.prefill_tps && g.prefill_tps.p50 != null ? g.prefill_tps.p50 : null },
    eff: { group: 'throughput', unit: 't/s', pick: g => g.effective_tps && g.effective_tps.p50 != null ? g.effective_tps.p50 : null },
    inputTpm: { group: 'throughput', unit: 'tpm', pick: g => g.input_tpm != null ? g.input_tpm : null },
    outputTpm: { group: 'throughput', unit: 'tpm', pick: g => g.output_tpm != null ? g.output_tpm : null },
  };
  const TREND_IDS = Object.keys(CHART_META);

  function fmtAxis(v) {
    if (v >= 10000) return Math.round(v / 1000) + 'k';
    if (v >= 1000) return (v / 1000).toFixed(1) + 'k';
    if (v >= 100) return String(Math.round(v));
    if (v >= 10) return v.toFixed(0);
    if (v >= 1) return v.toFixed(1);
    return v.toFixed(2);
  }
  function fmtTipVal(v, unit) {
    if (v == null || !isFinite(v)) return '-';
    const s = (unit === 'ms' || v >= 100) ? Math.round(v) : v.toFixed(1);
    return s + ' ' + unit;
  }
  function niceCeil(v) {
    if (!(v > 0) || !isFinite(v)) return 1;
    const p = Math.pow(10, Math.floor(Math.log10(v)));
    const n = v / p;
    const f = n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10;
    return f * p;
  }
  function fmtClock(t) {
    const d = new Date(t * 1000);
    const pad = n => String(n).padStart(2, '0');
    return pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds());
  }

  class TrendChart {
    constructor(id, reg, hover, repaintGroup) {
      this.id = id;
      this.meta = CHART_META[id];
      this.reg = reg;
      this.hover = hover;
      this.repaintGroup = repaintGroup;
      this.M = { t: 14, r: 18, b: 26, l: 52 };
      this.lastHistory = [];
      this.surface = document.getElementById(id + 'Chart');
      this.svg = document.createElementNS(SVG_NS, 'svg');
      this.surface.appendChild(this.svg);
      this.tip = document.createElement('div');
      this.tip.className = 'trend-tip';
      this.surface.appendChild(this.tip);
      this.surface.addEventListener('mousemove', e => this.onHover(e.clientX));
      this.surface.addEventListener('mouseleave', () => this.setHover(null));
      this.surface.addEventListener('touchstart', e => { if (e.touches[0]) this.onHover(e.touches[0].clientX); }, { passive: true });
      this.surface.addEventListener('touchmove', e => { if (e.touches[0]) this.onHover(e.touches[0].clientX); }, { passive: true });
      this.surface.addEventListener('touchend', () => this.setHover(null));
      if (window.ResizeObserver) new ResizeObserver(() => this.paint(this.lastHistory)).observe(this.surface);
    }

    draw(history) {
      this.lastHistory = history || [];
      this.paint(this.lastHistory);
    }

    onHover(clientX) {
      const h = this.lastHistory;
      const rect = this.surface.getBoundingClientRect();
      const x = clientX - rect.left;
      if (h.length < 2 || x < this.M.l || x > rect.width - this.M.r) { this.setHover(null); return; }
      // 改为索引定位：根据 X 坐标计算最接近的数据点索引
      const pw = rect.width - this.M.l - this.M.r;
      const idx = Math.round((x - this.M.l) / pw * (h.length - 1));
      const best = Math.max(0, Math.min(h.length - 1, idx));
      this.setHover(best);
    }

    setHover(i) {
      if (this.hover[this.meta.group] === i) return;
      this.hover[this.meta.group] = i;
      this.repaintGroup(this.meta.group);
    }

    seriesStyle(idx) {
      if (idx < SERIES_COLORS.length) return { color: SERIES_COLORS[idx], dash: '' };
      return { color: '#94a3b8', dash: EXTRA_DASHES[(idx - SERIES_COLORS.length) % EXTRA_DASHES.length] };
    }

    paint(h) {
      const rect = this.surface.getBoundingClientRect();
      const W = Math.floor(rect.width), H = Math.floor(rect.height);
      this.svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
      this.tip.style.display = 'none';
      if (W < 60) { this.svg.innerHTML = ''; return; }

      const M = this.M;
      const pw = W - M.l - M.r, ph = H - M.t - M.b;

      // 采样点不足 2 个：不显示坐标轴，只给等待提示
      if (!h || h.length < 2) {
        this.svg.innerHTML = '<text x="' + (W / 2) + '" y="' + (H / 2)
          + '" text-anchor="middle" class="trend-empty">等待正式压测数据…</text>';
        return;
      }

      // 改为索引轴：数据点按数组下标均匀分布，避免时间间隔造成断层
      const n = h.length;
      const xOf = i => M.l + (n <= 1 ? 0 : i / (n - 1) * pw);

      const visibleKeys = this.reg.order.filter(k => !this.reg.hidden.has(k));
      const series = visibleKeys.map(key => {
        const slot = this.reg.order.indexOf(key);
        const st = this.seriesStyle(slot);
        const vals = h.map(p => {
          const g = p.data && p.data[key];
          return g ? this.meta.pick(g) : null;
        });
        // 改为索引定位：使用数组下标而非时间戳
        const points = vals.map((v, i) => v == null || !isFinite(v) ? null : { x: xOf(i), y: v })
          .filter(Boolean);
        return { key, color: st.color, dash: st.dash, vals, points };
      });
      const allVals = series.flatMap(s => s.points.map(p => p.y));
      const allHidden = visibleKeys.length === 0;
      const maxV = niceCeil((allVals.length ? Math.max.apply(null, allVals) : 1) * 1.12);
      const yOf = v => M.t + ph - Math.max(0, Math.min(v, maxV)) / maxV * ph;
      const baseY = M.t + ph;

      let s = '';
      // 横向网格 + y 刻度（0 基线颜色加深）
      s += '<defs>';
      series.forEach((ser, i) => {
        s += '<linearGradient id="grad-' + this.id + '-' + i + '" x1="0" y1="0" x2="0" y2="1">'
          + '<stop offset="0%" stop-color="' + ser.color + '" stop-opacity=".14"/>'
          + '<stop offset="100%" stop-color="' + ser.color + '" stop-opacity="0"/>'
          + '</linearGradient>';
      });
      s += '</defs>';
      for (let f = 0; f <= 1.0001; f += 0.25) {
        const val = maxV * f;
        const y = (baseY - ph * f).toFixed(1);
        const grid = f === 0 ? '#cbd5e1' : '#edf1f6';
        s += '<line x1="' + M.l + '" y1="' + y + '" x2="' + (W - M.r) + '" y2="' + y
          + '" stroke="' + grid + '" stroke-width="1"/>';
        s += '<text x="' + (M.l - 8) + '" y="' + (+y + 3.5).toFixed(1)
          + '" text-anchor="end" font-size="10" fill="#94a3b8" style="font-variant-numeric:tabular-nums">'
          + fmtAxis(val) + '</text>';
      }
      // x 轴时间刻度（5 个）：按索引均匀分布，显示对应时间
      for (let i = 0; i <= 4; i++) {
        const dataIdx = Math.round(i / 4 * (n - 1));
        const x = xOf(dataIdx);
        const anchor = i === 0 ? 'start' : i === 4 ? 'end' : 'middle';
        s += '<text x="' + x.toFixed(1) + '" y="' + (H - 8) + '" text-anchor="' + anchor
          + '" font-size="10" fill="#94a3b8" style="font-variant-numeric:tabular-nums">'
          + fmtClock(h[dataIdx].t) + '</text>';
      }

      // 没有可见数据时的提示（序列全隐藏 / 尚无成功采样）
      if (allHidden || !allVals.length) {
        const msg = allHidden ? '序列已全部隐藏，点击上方图例恢复' : '数据采集中…';
        this.svg.innerHTML = s + '<text x="' + (W / 2) + '" y="' + (M.t + ph / 2)
          + '" text-anchor="middle" class="trend-empty">' + msg + '</text>';
        return;
      }

      // 面积 + 折线（null 点断开为多段，缺数据不会被错误连线）
      series.forEach((ser, i) => {
        const segments = [];
        let seg = [];
        ser.vals.forEach((v, idx) => {
          if (v == null || !isFinite(v)) { if (seg.length) { segments.push(seg); seg = []; } }
          // 改为索引定位
          else seg.push({ x: xOf(idx), y: yOf(v), v });
        });
        if (seg.length) segments.push(seg);
        segments.forEach(pts => {
          if (!pts.length) return;
          const line = pts.map((p, j) => (j ? 'L' : 'M') + p.x.toFixed(1) + ',' + p.y.toFixed(1)).join('');
          if (pts.length >= 2) {
            const area = 'M' + pts[0].x.toFixed(1) + ',' + baseY + ' '
              + pts.map(p => 'L' + p.x.toFixed(1) + ',' + p.y.toFixed(1)).join(' ')
              + ' L' + pts[pts.length - 1].x.toFixed(1) + ',' + baseY + ' Z';
            s += '<path d="' + area + '" fill="url(#grad-' + this.id + '-' + i + ')"/>';
          }
          s += '<path d="' + line + '" fill="none" stroke="' + ser.color + '" stroke-width="2"'
            + ' stroke-linejoin="round" stroke-linecap="round"'
            + (ser.dash ? ' stroke-dasharray="' + ser.dash + '"' : '') + '/>';
        });
      });

      // 悬停十字线（同页签两图联动）
      const hi = this.hover[this.meta.group];
      if (hi != null && hi >= 0 && hi < h.length) {
        const cx = xOf(hi);
        s += '<line x1="' + cx.toFixed(1) + '" y1="' + M.t + '" x2="' + cx.toFixed(1) + '" y2="' + baseY
          + '" stroke="#94a3b8" stroke-width="1" stroke-dasharray="3 3"/>';
        const rows = [];
        series.forEach(ser => {
          const v = ser.vals[hi];
          if (v == null || !isFinite(v)) return;
          s += '<circle cx="' + cx.toFixed(1) + '" cy="' + yOf(v).toFixed(1) + '" r="3.5" fill="#fff" stroke="'
            + ser.color + '" stroke-width="2"/>';
          rows.push('<div class="tip-row"><span class="tip-dot" style="background:' + ser.color + '"></span>'
            + '<span>' + escHtml(ser.key) + '</span><span class="tip-val">' + fmtTipVal(v, this.meta.unit) + '</span></div>');
        });
        if (rows.length) {
          this.tip.innerHTML = '<div class="tip-time">' + fmtClock(h[hi].t) + '</div>' + rows.join('');
          this.tip.style.display = 'block';
          // 防止 tooltip 溢出右边界
          const tipW = this.tip.offsetWidth || 160;
          this.tip.style.left = (cx + 12 + tipW > W ? Math.max(4, cx - tipW - 12) : cx + 12) + 'px';
          this.tip.style.top = '6px';
        }
      }

      this.svg.innerHTML = s;
    }
  }

  function create() {
    injectCss();
    // 每次 create 独立一套颜色槽位注册与悬停状态（实时页 / 快照页互不干扰）
    const reg = { order: [], hidden: new Set() };
    const hover = { latency: null, throughput: null };
    const charts = {};
    let lastHistory = [];
    TREND_IDS.forEach(id => {
      const surface = document.getElementById(id + 'Chart');
      if (surface) charts[id] = new TrendChart(id, reg, hover, group => {
        TREND_IDS.filter(cid => CHART_META[cid].group === group)
          .forEach(cid => charts[cid] && charts[cid].paint(charts[cid].lastHistory));
      });
    });

    function renderLegends() {
      ['latency', 'throughput'].forEach(tab => {
        const el = document.getElementById(tab + 'Legend');
        if (!el) return;
        if (!reg.order.length) {
          el.innerHTML = '<span class="lg-empty">正式压测开始后按供应商/模型显示，点击图例可临时隐藏</span>';
          return;
        }
        el.innerHTML = reg.order.map((k, i) => {
          const st = charts[TREND_IDS[0]] ? charts[TREND_IDS[0]].seriesStyle(i) : { color: '#94a3b8', dash: '' };
          const off = reg.hidden.has(k) ? ' off' : '';
          const dotStyle = st.dash
            ? 'background:repeating-linear-gradient(90deg,' + st.color + ' 0 5px,transparent 5px 9px);'
            : 'background:' + st.color + ';';
          return '<span class="lg-item' + off + '" data-key="' + escHtml(k) + '">'
            + '<span class="lg-dot" style="' + dotStyle + '"></span>' + escHtml(k) + '</span>';
        }).join('');
      });
    }

    function draw(history) {
      TREND_IDS.forEach(id => charts[id] && charts[id].draw(history));
    }

    // 图例点击：隐藏 / 恢复某个供应商/模型
    ['latency', 'throughput'].forEach(tab => {
      const el = document.getElementById(tab + 'Legend');
      if (!el) return;
      el.addEventListener('click', e => {
        const item = e.target.closest('.lg-item');
        if (!item) return;
        const k = item.dataset.key;
        if (reg.hidden.has(k)) reg.hidden.delete(k);
        else reg.hidden.add(k);
        renderLegends();
        draw(lastHistory);
      });
    });

    let resizeTimer = null;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => draw(lastHistory), 150);
    });

    renderLegends();
    draw([]);

    return {
      // 每帧数据：注册新分组、刷新图例、按给定 history 重绘四张图
      update(keys, history) {
        (keys || []).forEach(k => { if (!reg.order.includes(k)) reg.order.push(k); });
        lastHistory = history || [];
        renderLegends();
        draw(lastHistory);
      },
      // 新任务：清空颜色槽位与悬停态
      reset() {
        reg.order = [];
        reg.hidden.clear();
        hover.latency = null;
        hover.throughput = null;
        lastHistory = [];
        renderLegends();
        draw([]);
      },
      // 容器从 display:none 切回可见时按当前数据重绘
      redraw() {
        draw(lastHistory);
      },
    };
  }

  window.TrendCharts = { create, CHART_META, TREND_IDS };
})();
