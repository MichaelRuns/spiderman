interface PlaybackControlsProps {
  speedMs: number;
  onChange: (ms: number) => void;
  currentIndex: number;
  totalSteps: number;
}

export function PlaybackControls({ speedMs, onChange, currentIndex, totalSteps }: PlaybackControlsProps) {
  return (
    <div className="hyperparams">
      <label>
        Playback speed: {speedMs <= 0 ? "instant" : `${speedMs}ms/step`}
        <input
          type="range"
          min={0}
          max={600}
          step={20}
          value={speedMs}
          onChange={(e) => onChange(Number(e.target.value))}
        />
      </label>
      <span className="playback-position">
        step {Math.max(currentIndex, 0)} / {Math.max(totalSteps - 1, 0)}
      </span>
    </div>
  );
}
