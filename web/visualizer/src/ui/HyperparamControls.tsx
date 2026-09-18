export interface Hyperparams {
  maxNewTokens: number;
  temperature: number;
  topK: number;
  /** Floor on how long each token takes to appear in the UI — real compute (often a few ms) is paced up to this, so generation stays watchable. Doesn't affect the measured perf stats, which reflect real compute time. */
  minMsPerToken: number;
}

interface HyperparamControlsProps {
  value: Hyperparams;
  onChange: (value: Hyperparams) => void;
  maxAllowedNewTokens: number;
  disabled: boolean;
}

export function HyperparamControls({ value, onChange, maxAllowedNewTokens, disabled }: HyperparamControlsProps) {
  const set = <K extends keyof Hyperparams>(key: K, v: Hyperparams[K]) => onChange({ ...value, [key]: v });

  return (
    <div className="hyperparams">
      <label>
        Max new tokens: {value.maxNewTokens}
        <input
          type="range"
          min={1}
          max={maxAllowedNewTokens}
          value={Math.min(value.maxNewTokens, maxAllowedNewTokens)}
          onChange={(e) => set("maxNewTokens", Number(e.target.value))}
          disabled={disabled}
        />
      </label>
      <label>
        Temperature: {value.temperature.toFixed(2)}
        <input
          type="range"
          min={0}
          max={2}
          step={0.05}
          value={value.temperature}
          onChange={(e) => set("temperature", Number(e.target.value))}
          disabled={disabled}
        />
      </label>
      <label>
        Top-K: {value.topK === 0 ? "off" : value.topK}
        <input
          type="range"
          min={0}
          max={50}
          value={value.topK}
          onChange={(e) => set("topK", Number(e.target.value))}
          disabled={disabled}
        />
      </label>
      <label>
        Min ms/token: {value.minMsPerToken === 0 ? "off (as fast as possible)" : value.minMsPerToken}
        <input
          type="range"
          min={0}
          max={300}
          step={10}
          value={value.minMsPerToken}
          onChange={(e) => set("minMsPerToken", Number(e.target.value))}
          disabled={disabled}
        />
      </label>
    </div>
  );
}
