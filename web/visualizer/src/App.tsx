import { useEffect, useMemo, useState } from "react";
import { generate } from "./inference/generate.js";
import { loadModel, type LoadedModel } from "./inference/loadModel.js";
import type { StepSnapshot } from "./inference/snapshot.js";
import { ArchitectureCanvas } from "./scene/ArchitectureCanvas.js";
import { HyperparamControls, type Hyperparams } from "./ui/HyperparamControls.js";
import { NextTokenRanking } from "./ui/NextTokenRanking.js";
import { PerfStatsPanel } from "./ui/PerfStatsPanel.js";
import { initialPerfStats, updatePerfStats, type PerfStats } from "./ui/perfStats.js";
import { UnifiedInput } from "./ui/UnifiedInput.js";
import { useStepSelection } from "./ui/useStepSelection.js";

type LoadState = { status: "loading" } | { status: "ready"; loaded: LoadedModel } | { status: "error"; message: string };

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function App() {
  const [loadState, setLoadState] = useState<LoadState>({ status: "loading" });
  const [text, setText] = useState("Once upon a time");
  const [hyperparams, setHyperparams] = useState<Hyperparams>({
    maxNewTokens: 60,
    temperature: 0.8,
    topK: 20,
    minMsPerToken: 40,
  });
  const [isGenerating, setIsGenerating] = useState(false);
  const [perfStats, setPerfStats] = useState<PerfStats>(initialPerfStats);
  const [steps, setSteps] = useState<StepSnapshot[]>([]);
  const [selectedLayer, setSelectedLayer] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const selection = useStepSelection(steps);

  useEffect(() => {
    loadModel(import.meta.env.BASE_URL)
      .then((loaded) => setLoadState({ status: "ready", loaded }))
      .catch((err: unknown) => setLoadState({ status: "error", message: String(err) }));
  }, []);

  const stopTokenIds = useMemo(() => {
    if (loadState.status !== "ready") return new Set<number>();
    const { tokenizer, manifest } = loadState.loaded;
    return new Set(manifest.specialTokens.flatMap((token) => tokenizer.encode(token)));
  }, [loadState]);

  const currentTokenCount = useMemo(() => {
    if (loadState.status !== "ready" || text.length === 0) return 0;
    return loadState.loaded.tokenizer.encode(text).length;
  }, [loadState, text]);

  const maxAllowedNewTokens =
    loadState.status === "ready"
      ? Math.max(1, loadState.loaded.manifest.config.contextLength - currentTokenCount - 1)
      : 256;

  function handleTextChange(next: string) {
    setText(next);
    // Only real user edits reach here (the textarea is disabled mid-generation, and our
    // own token-appending during generation sets `text` directly, not via this handler) —
    // an edit invalidates the "trailing tokens = last generation run" assumption the
    // clickable chips rely on, so drop them rather than let stale chips point at the wrong step.
    setSteps([]);
  }

  async function handleGenerate() {
    if (loadState.status !== "ready") return;
    const { model, tokenizer, manifest } = loadState.loaded;

    setError(null);
    setPerfStats(initialPerfStats);
    setSteps([]);
    setIsGenerating(true);

    // Tracked locally, not via the `perfStats` state variable — React state stays frozen
    // at its value from when this closure started, so reading it back across iterations
    // of this same loop would always see the pre-generation value, not the running total.
    let runningPerf = initialPerfStats;

    try {
      for await (const step of generate(model, tokenizer, text, hyperparams, stopTokenIds)) {
        runningPerf = updatePerfStats(runningPerf, step, manifest.config.dModel, manifest.config.numLayers);

        // Real compute is often a few ms/token — far too fast to watch. Pace the *display*
        // up to `minMsPerToken` without touching the perf stats above, which already
        // captured the real `step.latencyMs` before this delay.
        const remaining = hyperparams.minMsPerToken - step.latencyMs;
        if (remaining > 0) await sleep(remaining);

        setPerfStats(runningPerf);
        setText((prev) => prev + step.tokenText);
        setSteps((prev) => [...prev, step.snapshot]);

        if (step.done) break;
      }
    } catch (err) {
      setError(String(err));
    } finally {
      setIsGenerating(false);
    }
  }

  if (loadState.status === "loading") {
    return (
      <main className="app">
        <h1>spiderman</h1>
        <p>Loading model weights…</p>
      </main>
    );
  }

  if (loadState.status === "error") {
    return (
      <main className="app">
        <h1>spiderman</h1>
        <p className="error">Failed to load model: {loadState.message}</p>
      </main>
    );
  }

  const { tokenizer, manifest } = loadState.loaded;
  const dK = manifest.config.dModel / manifest.config.numHeads;

  return (
    <main className="app">
      <header>
        <h1>spiderman</h1>
        <p className="subtitle">
          A transparent-frame LLM — {manifest.config.numLayers} layers, {manifest.config.dModel}d,{" "}
          {manifest.config.numHeads} heads, {manifest.config.vocabSize.toLocaleString()} vocab. The diagram below is
          driven by the real forward pass, not a static illustration.
        </p>
      </header>

      <UnifiedInput
        text={text}
        onChange={handleTextChange}
        onGenerate={handleGenerate}
        isGenerating={isGenerating}
        tokenizer={tokenizer}
        steps={steps}
        selectedIndex={selection.currentIndex}
        isFollowingLatest={selection.isFollowingLatest}
        onSelect={selection.select}
        onFollowLatest={selection.followLatest}
      />
      <HyperparamControls
        value={hyperparams}
        onChange={setHyperparams}
        maxAllowedNewTokens={maxAllowedNewTokens}
        disabled={isGenerating}
      />

      {error && <p className="error">{error}</p>}

      <section>
        <h2>Next-token probabilities</h2>
        <p className="section-hint">
          Ranked softmax output at the selected step — the highlighted row is the token actually sampled.
        </p>
        <NextTokenRanking step={selection.currentStep} tokenizer={tokenizer} />
      </section>

      <section>
        <h2>Inside the model</h2>
        <p className="section-hint">Click a token above to inspect that step; otherwise this follows the latest one.</p>
        <ArchitectureCanvas
          config={{
            numLayers: manifest.config.numLayers,
            numHeads: manifest.config.numHeads,
            dK,
            dModel: manifest.config.dModel,
            contextLength: manifest.config.contextLength,
          }}
          step={selection.currentStep}
          selectedLayer={selectedLayer}
          onSelectLayer={setSelectedLayer}
        />
      </section>

      <section>
        <h2>Perf</h2>
        <PerfStatsPanel stats={perfStats} />
      </section>
    </main>
  );
}
