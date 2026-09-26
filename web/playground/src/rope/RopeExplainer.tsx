import { NDArray, RoPE } from "@spiderman/nn";
import { useEffect, useMemo, useRef, useState } from "react";
import { drawVectorPair } from "./vectorCanvas.js";

const MAX_SEQ_LEN = 32;

function defaultX(dK: number): number[] {
  // Each pair starts as the unit vector (1, 0) — after rotation it points
  // directly at its own angle, so the plot *is* the angle, not just related to it.
  return Array.from({ length: dK }, (_, i) => (i % 2 === 0 ? 1 : 0));
}

function PairCard({
  index,
  before,
  after,
  thetaI,
  angleRad,
  cos,
  sin,
}: {
  index: number;
  before: [number, number];
  after: [number, number];
  thetaI: number;
  angleRad: number;
  cos: number;
  sin: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (canvasRef.current) drawVectorPair(canvasRef.current, before, after, angleRad);
  }, [before, after, angleRad]);

  const degrees = (angleRad * 180) / Math.PI;

  return (
    <div className="pair-card">
      <h3>pair {index}</h3>
      <canvas ref={canvasRef} className="pair-canvas" />
      <dl className="pair-numbers">
        <div>
          <dt>
            (x₍{2 * index}₎, x₍{2 * index + 1}₎)
          </dt>
          <dd>
            ({before[0].toFixed(2)}, {before[1].toFixed(2)})
          </dd>
        </div>
        <div>
          <dt>θᵢ = θ⁻²ⁱ/ᵈ</dt>
          <dd>{thetaI.toExponential(3)}</dd>
        </div>
        <div>
          <dt>angle = m·θᵢ</dt>
          <dd>
            {angleRad.toFixed(4)} rad ({degrees.toFixed(1)}°)
          </dd>
        </div>
        <div>
          <dt>cos, sin</dt>
          <dd>
            {cos.toFixed(4)}, {sin.toFixed(4)}
          </dd>
        </div>
        <div>
          <dt>after = (x·cos−y·sin, x·sin+y·cos)</dt>
          <dd>
            ({after[0].toFixed(2)}, {after[1].toFixed(2)})
          </dd>
        </div>
      </dl>
    </div>
  );
}

export function RopeExplainer() {
  const [dK, setDK] = useState(8);
  const [theta, setTheta] = useState(10000);
  const [position, setPosition] = useState(2);
  const [x, setX] = useState<number[]>(() => defaultX(8));

  const rope = useMemo(() => new RoPE(dK, MAX_SEQ_LEN, theta), [dK, theta]);

  const { rotated, angles } = useMemo(() => {
    const out = rope.forward(NDArray.fromNested([x]), NDArray.fromNested([position]));
    return { rotated: Array.from(out.data), angles: rope.anglesAt(position) };
  }, [rope, x, position]);

  function handleDKChange(newDK: number) {
    setDK(newDK);
    setX(defaultX(newDK));
  }

  function handleXChange(i: number, value: number) {
    setX((prev) => prev.map((v, j) => (j === i ? value : v)));
  }

  const half = dK / 2;

  return (
    <div className="explainer">
      <p className="explainer-intro">
        RoPE doesn't add a separate positional vector — it <em>rotates</em> each adjacent pair of a
        query/key vector by an angle that depends on the pair's index and the token's position. Every
        pair rotates at its own frequency <code>θᵢ = θ⁻²ⁱ/ᵈ</code>, so the first pair (i=0) rotates
        fastest (frequency 1) and later pairs rotate ever more slowly — the same trick behind sinusoidal
        embeddings, just applied as a rotation instead of an addition.
      </p>

      <div className="controls">
        <label>
          d_k
          <select value={dK} onChange={(e) => handleDKChange(Number(e.target.value))}>
            <option value={4}>4</option>
            <option value={8}>8</option>
          </select>
        </label>
        <label>
          θ (theta base)
          <input type="number" value={theta} min={2} step={100} onChange={(e) => setTheta(Number(e.target.value))} />
        </label>
        <label>
          position m: {position}
          <input
            type="range"
            min={0}
            max={MAX_SEQ_LEN - 1}
            value={position}
            onChange={(e) => setPosition(Number(e.target.value))}
          />
        </label>
      </div>

      <div className="x-inputs">
        <span className="x-inputs-label">input vector x:</span>
        {x.map((v, i) => (
          <label key={i} className="x-input">
            x₍{i}₎
            <input type="number" value={v} step={0.5} onChange={(e) => handleXChange(i, Number(e.target.value))} />
          </label>
        ))}
      </div>

      <div className="pair-grid">
        {Array.from({ length: half }, (_, i) => (
          <PairCard
            key={i}
            index={i}
            before={[x[2 * i]!, x[2 * i + 1]!]}
            after={[rotated[2 * i]!, rotated[2 * i + 1]!]}
            thetaI={angles.thetaI[i]!}
            angleRad={angles.angleRad[i]!}
            cos={angles.cos[i]!}
            sin={angles.sin[i]!}
          />
        ))}
      </div>
    </div>
  );
}
