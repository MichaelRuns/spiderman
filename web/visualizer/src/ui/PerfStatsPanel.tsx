import type { PerfStats } from "./perfStats.js";

function fmtMs(ms: number | null): string {
  return ms === null ? "—" : `${ms.toFixed(1)}ms`;
}

function fmtBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)}MB`;
}

export function PerfStatsPanel({ stats }: { stats: PerfStats }) {
  return (
    <dl className="perf-stats">
      <div>
        <dt>Prefill</dt>
        <dd>{fmtMs(stats.prefillMs)}</dd>
      </div>
      <div>
        <dt>Current token</dt>
        <dd>{fmtMs(stats.currentMs)}</dd>
      </div>
      <div>
        <dt>Min time/token</dt>
        <dd>{fmtMs(stats.minMs)}</dd>
      </div>
      <div>
        <dt>Avg time/token</dt>
        <dd>{fmtMs(stats.avgMs)}</dd>
      </div>
      <div>
        <dt>Tokens/sec</dt>
        <dd>{stats.tokensPerSec === null ? "—" : stats.tokensPerSec.toFixed(1)}</dd>
      </div>
      <div>
        <dt>Tokens generated</dt>
        <dd>{stats.tokensGenerated}</dd>
      </div>
      <div>
        <dt>KV cache size</dt>
        <dd>{fmtBytes(stats.kvCacheBytes)}</dd>
      </div>
    </dl>
  );
}
