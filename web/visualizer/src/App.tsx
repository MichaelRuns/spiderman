import { useEffect, useMemo, useState } from "react";
import { generate } from "./inference/generate.js";
import { loadModel, type LoadedModel } from "./inference/loadModel.js";
import type { StepSnapshot } from "./inference/snapshot.js";
import { HyperparamControls, type Hyperparams } from "./ui/HyperparamControls.js";
import { PerfStatsPanel } from "./ui/PerfStatsPanel.js";
import { PlaybackControls } from "./ui/PlaybackControls.js";
import { initialPerfStats, updatePerfStats, type PerfStats } from "./ui/perfStats.js";
import { PromptInput } from "./ui/PromptInput.js";
import { usePlayback } from "./ui/usePlayback.js";
import { Scene } from "./scene/Scene.js";

type LoadState = { status: "loading" } | { status: "ready"; loaded: LoadedModel } | { status: "error"; message: string };

export function App() {
  const [loadState, setLoadState] = useState<LoadState>({ status: "loading" });
  const [prompt, setPrompt] = useState("Once upon a time");
  const [hyperparams, setHyperparams] = useState<Hyperparams>({ maxNewTokens: 60, temperature: 0.8, topK: 20 });
  const [generatedText, setGeneratedText] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [perfStats, setPerfStats] = useState<PerfStats>(initialPerfStats);
  const [steps, setSteps] = useState<StepSnapshot[]>([]);
  const [error, setError] = useState<string | null>(null);

  const playback = usePlayback(steps);

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

  const maxAllowedNewTokens =
    loadState.status === "ready" ? loadState.loaded.manifest.config.contextLength - 1 : 256;

  async function handleGenerate() {
    if (loadState.status !== "ready") return;
    const { model, tokenizer, manifest } = loadState.loaded;

    setError(null);
    setGeneratedText(prompt);
    setPerfStats(initialPerfStats);
    setSteps([]);
    setIsGenerating(true);

    // Real generation can produce well over 100 tokens/sec on this small model — far
    // faster than React (or the 3D scene) needs to re-render. Batch state flushes to a
    // fixed cadence instead of committing once per token, so render pressure stays
    // bounded regardless of how fast the actual compute runs.
    let pendingText = prompt;
    let pendingPerf = initialPerfStats;
    let pendingSteps: StepSnapshot[] = [];
    let lastFlush = performance.now();
    const FLUSH_INTERVAL_MS = 50;

    const flush = () => {
      setGeneratedText(pendingText);
      setPerfStats(pendingPerf);
      const toAppend = pendingSteps;
      setSteps((prev) => [...prev, ...toAppend]);
      pendingSteps = [];
      lastFlush = performance.now();
    };

    try {
      for await (const step of generate(model, tokenizer, prompt, hyperparams, stopTokenIds)) {
        pendingText += step.tokenText;
        pendingPerf = updatePerfStats(pendingPerf, step, manifest.config.dModel, manifest.config.numLayers);
        pendingSteps.push(step.snapshot);

        if (step.done || performance.now() - lastFlush >= FLUSH_INTERVAL_MS) {
          flush();
        }
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

  const { manifest } = loadState.loaded;
  const dK = manifest.config.dModel / manifest.config.numHeads;

  return (
    <main className="app">
      <header>
        <h1>spiderman</h1>
        <p className="subtitle">
          A transparent-frame LLM — {manifest.config.numLayers} layers, {manifest.config.dModel}d,{" "}
          {manifest.config.numHeads} heads, {manifest.config.vocabSize.toLocaleString()} vocab. Every layer below is
          the real forward pass, not a diagram.
        </p>
      </header>

      <PromptInput value={prompt} onChange={setPrompt} onSubmit={handleGenerate} disabled={isGenerating} />
      <HyperparamControls
        value={hyperparams}
        onChange={setHyperparams}
        maxAllowedNewTokens={maxAllowedNewTokens}
        disabled={isGenerating}
      />

      {error && <p className="error">{error}</p>}

      <section>
        <h2>Inside the model</h2>
        <PlaybackControls
          speedMs={playback.speedMs}
          onChange={playback.setSpeedMs}
          currentIndex={playback.currentIndex}
          totalSteps={playback.totalSteps}
        />
        <Scene
          step={playback.currentStep}
          numLayers={manifest.config.numLayers}
          numHeads={manifest.config.numHeads}
          dK={dK}
          contextLength={manifest.config.contextLength}
        />
      </section>

      <section className="output">
        <h2>Output</h2>
        <pre>{generatedText || "…"}</pre>
      </section>

      <section>
        <h2>Perf</h2>
        <PerfStatsPanel stats={perfStats} />
      </section>
    </main>
  );
}
