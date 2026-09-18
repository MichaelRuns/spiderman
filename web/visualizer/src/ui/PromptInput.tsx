interface PromptInputProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  disabled: boolean;
}

export function PromptInput({ value, onChange, onSubmit, disabled }: PromptInputProps) {
  return (
    <div className="prompt-input">
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Once upon a time..."
        rows={3}
        disabled={disabled}
      />
      <button
        onClick={onSubmit}
        disabled={disabled || value.trim().length === 0}
        type="button"
      >
        {disabled ? "Generating…" : "Generate"}
      </button>
    </div>
  );
}
