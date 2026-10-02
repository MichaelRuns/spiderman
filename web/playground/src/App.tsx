import { useState } from "react";
import { AttentionStepThrough } from "./attention/AttentionStepThrough.js";
import { BpeExplainer } from "./bpe/BpeExplainer.js";
import { RopeExplainer } from "./rope/RopeExplainer.js";

type Tab = "bpe" | "rope" | "attention";

export function App() {
  const [tab, setTab] = useState<Tab>("bpe");

  return (
    <main className="app">
      <header>
        <h1>module playground</h1>
        <p className="subtitle">
          How individual pieces of the model work, independent of running the full thing — every number
          computed by the real <code>@spiderman/nn</code> code, not a reimplementation for illustration.
        </p>
      </header>

      <div className="tabs">
        <button
          type="button"
          className={tab === "bpe" ? "tab active" : "tab"}
          onClick={() => setTab("bpe")}
        >
          BPE
        </button>
        <button type="button" className={tab === "rope" ? "tab active" : "tab"} onClick={() => setTab("rope")}>
          RoPE
        </button>
        <button
          type="button"
          className={tab === "attention" ? "tab active" : "tab"}
          onClick={() => setTab("attention")}
        >
          Attention
        </button>
      </div>

      {tab === "bpe" ? <BpeExplainer /> : tab === "rope" ? <RopeExplainer /> : <AttentionStepThrough />}
    </main>
  );
}
