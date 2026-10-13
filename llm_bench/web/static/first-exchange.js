/*
 * 首请求校验公共组件（每个模型首个请求的完整请求 / 返回 Body）
 *
 * 实时压测页（dashboard.html）与离线页面快照（report.html）共用本文件：
 * 列表渲染、折叠展开、复制按钮的交互完全一致；仅「展开时如何取完整 body」不同，
 * 由调用方提供 resolve(key, meta)，可同步返回 exchange，也可返回 Promise。
 * 快照生成时由 snapshot.py 把本文件整体内联进 report.html，保持单文件离线可看。
 *
 * DOM 约定：
 *   卡片      #firstExchangeCard（含 .card-header 折叠头 / #firstExchangeCount 计数）
 *   列表容器  #firstExchangeList
 *
 * 用法：
 *   const ui = FirstExchange.mount({
 *     resolve(key, meta) { return exchange; }  // 或 fetch(...).then(...)
 *   });
 *   ui.setIndex({ "provider/model": meta, ... });  // 轻量索引，不含 body
 *   ui.reset();                                     // 新任务：清空展开态
 */
(function () {
  'use strict';

  const CSS = `
#firstExchangeCard > .card-header { cursor: pointer; user-select: none; }
#firstExchangeCard .fe-card-toggle {
  display: inline-block; width: 12px; color: var(--text-light, #64748b);
  font-size: 10px; transition: transform .15s;
}
#firstExchangeCard.open .fe-card-toggle { transform: rotate(90deg); }
#firstExchangeCard .card-body.collapsed { display: none; }
.fe-item { border: 1px solid var(--border, #e2e8f0); border-radius: 6px; margin-bottom: 8px; }
.fe-item:last-child { margin-bottom: 0; }
.fe-item-header {
  display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
  padding: 9px 12px; font-size: 13px; cursor: pointer;
}
.fe-item-header:hover { background: #f8fafc; }
.fe-item-toggle { color: var(--text-light, #64748b); font-size: 9px; width: 10px; transition: transform .15s; }
.fe-item.open .fe-item-toggle { transform: rotate(90deg); }
.fe-item-body { display: none; border-top: 1px solid var(--border, #e2e8f0); padding: 12px; }
.fe-item.open .fe-item-body { display: block; }
.fe-pane + .fe-pane { margin-top: 14px; }
.fe-pane-bar { display: flex; align-items: center; gap: 8px; font-size: 12px; font-weight: 600; margin-bottom: 6px; }
.fe-pane-bar .spacer { flex: 1; }
.fe-pane-bar .fe-url { font-weight: 400; font-family: ui-monospace, Menlo, Consolas, monospace; }
.fe-pre {
  background: #0f172a; color: #e2e8f0; padding: 12px; border-radius: 6px; margin: 0;
  font-size: 12px; line-height: 1.55; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  white-space: pre-wrap; word-break: break-word; max-height: 380px; overflow: auto;
}
.fe-pre.is-error { background: #2a1414; color: #fecaca; }
.fe-badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 500; }
.fe-badge-ok { background: #dcfce7; color: #16a34a; }
.fe-badge-err { background: #fee2e2; color: #dc2626; }
.fe-badge-idle { background: #f1f5f9; color: #64748b; }
.fe-copy {
  padding: 3px 10px; font-size: 12px; border: 1px solid var(--border, #e2e8f0);
  border-radius: 6px; background: #f1f5f9; color: var(--text, #1e293b); cursor: pointer;
}
.fe-copy:hover { background: #e2e8f0; }
.fe-toast {
  position: fixed; top: 20px; right: 20px; z-index: 2000; display: none;
  padding: 10px 18px; border-radius: 6px; color: #fff; font-size: 13px;
  box-shadow: 0 4px 12px rgba(0,0,0,.15);
}
.fe-toast.ok { background: #22c55e; }
.fe-toast.err { background: #ef4444; }
`;

  function injectCss() {
    if (document.querySelector('style[data-first-exchange]')) return;
    const style = document.createElement('style');
    style.dataset.firstExchange = '1';
    style.textContent = CSS;
    document.head.appendChild(style);
  }

  function escHtml(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, c => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
  }

  function itemHtml(key, meta) {
    const ok = !!meta.success;
    const errTitle = meta.error_message || meta.error_type || '';
    const badge = ok
      ? '<span class="fe-badge fe-badge-ok">成功</span>'
      : '<span class="fe-badge fe-badge-err" title="' + escHtml(errTitle) + '">失败</span>';
    const http = meta.http_status != null ? ('HTTP ' + meta.http_status) : (meta.error_type || '无响应');
    const mode = meta.streamed ? '流式' : '非流式';
    const respKind = meta.streamed ? '原始 SSE 报文' : '原始 JSON';
    return '<div class="fe-item" data-key="' + escHtml(key)
      + '" data-provider="' + escHtml(meta.provider || '')
      + '" data-model="' + escHtml(meta.model || '') + '">'
      + '<div class="fe-item-header">'
      + '<span class="fe-item-toggle">▶</span>'
      + '<strong>' + escHtml(key) + '</strong>'
      + '<span class="muted">' + escHtml(meta.phase_label || meta.phase || '') + '首请求</span>'
      + badge
      + '<span class="muted">' + escHtml(http) + ' · ' + escHtml(mode) + '</span>'
      + '</div>'
      + '<div class="fe-item-body">'
      + '<div class="fe-pane"><div class="fe-pane-bar"><span>请求 Body</span>'
      +   '<span class="spacer"></span><span class="fe-url muted"></span>'
      +   '<button type="button" class="fe-copy" data-pane="req">复制</button></div>'
      +   '<pre class="fe-pre" data-pane="req">加载中…</pre></div>'
      + '<div class="fe-pane"><div class="fe-pane-bar"><span>返回 Body</span>'
      +   '<span class="spacer"></span><span class="muted">' + respKind + '</span>'
      +   '<button type="button" class="fe-copy" data-pane="resp">复制</button></div>'
      +   '<pre class="fe-pre" data-pane="resp">加载中…</pre></div>'
      + '</div></div>';
  }

  function fillItem(item, ex) {
    const reqText = ex.request_body == null
      ? '（无请求体）'
      : JSON.stringify(ex.request_body, null, 2);
    let respText = ex.response_text || '';
    if (!respText) {
      respText = ex.error_message ? ('（无返回内容）\n' + ex.error_message) : '（无返回内容）';
    }
    const reqPre = item.querySelector('.fe-pre[data-pane="req"]');
    const respPre = item.querySelector('.fe-pre[data-pane="resp"]');
    reqPre.textContent = reqText;
    respPre.textContent = respText;
    respPre.classList.toggle('is-error', !ex.success);
    const urlEl = item.querySelector('.fe-url');
    if (urlEl) urlEl.textContent = ex.url || '';
  }

  function showToast(msg, ok) {
    let toast = document.querySelector('.fe-toast');
    if (!toast) {
      toast = document.createElement('div');
      toast.className = 'fe-toast';
      document.body.appendChild(toast);
    }
    toast.textContent = msg;
    toast.className = 'fe-toast ' + (ok ? 'ok' : 'err');
    toast.style.display = 'block';
    clearTimeout(toast._timer);
    toast._timer = setTimeout(() => { toast.style.display = 'none'; }, 3000);
  }

  function mount(options) {
    options = options || {};
    const resolve = options.resolve || function () { return null; };
    injectCss();

    const card = document.getElementById('firstExchangeCard');
    const list = document.getElementById('firstExchangeList');
    const countEl = document.getElementById('firstExchangeCount');
    const expanded = new Set();
    const cache = new Map();
    let sig = '';

    if (card) {
      card.querySelector('.card-header').addEventListener('click', () => {
        card.classList.toggle('open');
        const body = card.querySelector('.card-body');
        if (body) body.classList.toggle('collapsed', !card.classList.contains('open'));
      });
    }

    function showLoadingError(item, message) {
      item.querySelectorAll('.fe-pre').forEach(p => {
        p.textContent = '加载失败：' + message;
        p.classList.add('is-error');
      });
    }

    function findItem(key) {
      // 用 dataset 精确比较，避免 provider/model 名中的特殊字符破坏属性选择器
      return Array.from(list.querySelectorAll('.fe-item'))
        .find(el => el.dataset.key === key) || null;
    }

    function expand(key) {
      const item = findItem(key);
      if (!item) return;
      item.classList.add('open');
      if (cache.has(key)) {
        fillItem(item, cache.get(key));
        return;
      }
      let ret;
      try {
        ret = resolve(key, item.dataset);
      } catch (e) {
        showLoadingError(item, e.message);
        return;
      }
      if (ret && typeof ret.then === 'function') {
        ret.then(ex => {
          if (ex == null) throw new Error('无首请求留证数据');
          cache.set(key, ex);
          const cur = findItem(key);
          if (cur && cur.classList.contains('open')) fillItem(cur, ex);
        }).catch(e => {
          const cur = findItem(key);
          if (cur) showLoadingError(cur, e && e.message || String(e));
        });
      } else if (ret != null) {
        cache.set(key, ret);
        fillItem(item, ret);
      }
    }

    // 事件委托：行头折叠展开；复制按钮不触发折叠
    list.addEventListener('click', async e => {
      const copyBtn = e.target.closest('.fe-copy');
      if (copyBtn) {
        e.stopPropagation();
        const item = copyBtn.closest('.fe-item');
        const ex = item && cache.get(item.dataset.key);
        if (!ex) { showToast('请先展开该模型', false); return; }
        const text = copyBtn.dataset.pane === 'req'
          ? JSON.stringify(ex.request_body, null, 2)
          : (ex.response_text || ex.error_message || '');
        try {
          await navigator.clipboard.writeText(text == null ? '' : String(text));
          showToast('已复制' + (copyBtn.dataset.pane === 'req' ? '请求' : '返回') + ' Body', true);
        } catch (_) {
          showToast('复制失败，请手动选择文本复制', false);
        }
        return;
      }
      const header = e.target.closest('.fe-item-header');
      if (!header) return;
      const item = header.closest('.fe-item');
      const key = item.dataset.key;
      if (item.classList.contains('open')) {
        item.classList.remove('open');
        expanded.delete(key);
      } else {
        expanded.add(key);
        expand(key);
      }
    });

    function setIndex(index) {
      const keys = Object.keys(index || {});
      let fail = 0;
      keys.forEach(k => { if (!index[k].success) fail += 1; });
      if (countEl) {
        countEl.textContent = keys.length ? String(keys.length) : '0';
        countEl.className = 'fe-badge ' + (fail ? 'fe-badge-err' : (keys.length ? 'fe-badge-ok' : 'fe-badge-idle'));
      }

      // 指标帧每秒到达：索引内容未变化时不重建 DOM，避免打断展开/滚动
      const nextSig = JSON.stringify(index || {});
      if (nextSig === sig) return;
      sig = nextSig;

      if (!keys.length) {
        list.className = 'muted';
        list.style.fontSize = '13px';
        list.textContent = '任务启动后，将在此按模型展示首个请求（缓存预热 / 预热 / 正式压测中取最早一次），点击模型行展开';
        return;
      }
      list.className = '';
      list.style.fontSize = '';
      list.innerHTML = keys.map(k => itemHtml(k, index[k])).join('');
      // 重建后恢复此前已展开的模型（body 已缓存，不重复请求）
      expanded.forEach(k => expand(k));
    }

    function reset() {
      sig = '';
      expanded.clear();
      cache.clear();
    }

    return { setIndex, reset, expand };
  }

  window.FirstExchange = { mount };
})();
